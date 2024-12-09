import transformers, datasets
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import argparse
from cka import feature_space_linear_cka
import copy
import os
import pickle
import numpy as np
import torch
from deep_sim import deepDot, deepCKA, ContrastiveSim, ContrastiveSim_dis
from utils import SupConLoss, SupConLossDis
import torch.nn.functional as F


def create_dataset(split, device, args):
    with torch.no_grad():
        model = AutoModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True,
                                          output_attentions=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        dataset = load_dataset('xnli', 'all_languages')
        languages_data = [[] for _ in range(len(args.languages))]
        language_list = args.languages
        sentences = []
        num_embeddings = args.num_of_test_embeddings if split == 'test' else args.num_of_train_embeddings
        index_list = [dataset['test'][0]['hypothesis']['language'].index(lang) for lang in language_list]

        for i, example in enumerate(dataset[split]):
            if i >= num_embeddings:
                break
            premise_list = []
            if example['premise']['en'] not in sentences:
                premise_list.extend(example['premise'][language] for language in language_list)
                sentences.append(example['premise']['en'])

            if example['hypothesis']['translation'][index_list[1]] not in sentences:
                hypothesis_list = [example['hypothesis']['translation'][index] for index in index_list]
                sentences.append(hypothesis_list[0])

            p_encoded_list = [tokenizer(p, return_tensors='pt').to(device) for p in premise_list]
            h_encdoded_list = [tokenizer(h, return_tensors='pt').to(device) for h in hypothesis_list]

            p_output_list = [model(**p).hidden_states[1:] for p in p_encoded_list]
            h_output_list = [model(**h).hidden_states[1:] for h in h_encdoded_list]

            p_cls = []
            h_cls = []

            for p1 in p_output_list:
                p1 = [p.squeeze()[0] for p in p1]
                p1 = torch.stack(p1)
                p_cls.append(p1)

            for h1 in h_output_list:
                h1 = [h.squeeze()[0] for h in h1]
                h1 = torch.stack(h1)
                h_cls.append(h1)

            for j, p in enumerate(p_cls):
                languages_data[j].append(p.cpu().detach())

            for j, h in enumerate(h_cls):
                languages_data[j].append(h.cpu().detach())

        languages_data = [torch.stack(l) for l in languages_data]  # num_sentences x num_layers x out_dim

        dict = {'tensor': torch.stack(languages_data), 'sentences': sentences, 'indices': language_list}
        return dict  # tensor: num_languages x num_sentences x num_layers x out_dim


def create_dataset_faiss(split, device, args):
    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    def get_sentences():
        dataset = load_dataset('xnli', 'all_languages', split=f'{split}')
        language_list = args.languages
        index_list = [dataset[0]['hypothesis']['language'].index(lang) for lang in language_list]
        sentences = [[] for _ in range(len(args.languages))]
        num_embeddings = args.num_of_test_embeddings if split == 'test' else args.num_of_train_embeddings

        for i, example in enumerate(dataset):
            if i >= num_embeddings:
                break
            if example['premise']['en'] not in sentences[0]:
                for i in range(len(args.languages)):
                    sentences[i].append(example['premise'][language_list[i]])

            if example['hypothesis']['translation'][index_list[1]] not in sentences[0]:
                hypothesis_list = [example['hypothesis']['translation'][index] for index in index_list]
                for i in range(len(args.languages)):
                    sentences[i].append(hypothesis_list[i])

        return sentences  # num_languages x num_sentences x num_layers x out_dim

    sentences = get_sentences()
    sentences = [s[:args.num_of_test_embeddings if split == 'test' else args.num_of_train_embeddings] for s in
                 sentences]
    ds = datasets.Dataset.from_dict(
        {f'sentences_{args.languages[i]}': sentences[i] for i in range(len(args.languages))})

    def get_text_embeddings(example, layer_num, language):
        text = example[f'sentences_{language}']
        encoded = tokenizer(text, return_tensors='pt').to(device)
        output = model(**encoded).hidden_states[layer_num].squeeze()[0, :].cpu().detach()
        return output

    for i in range(1, 13, 1):
        for lang_ind, lang in enumerate(args.languages):
            ds = ds.map(lambda example: {f'{lang}_embed_{i}': get_text_embeddings(example, i, lang)})

    for i in range(1, 13, 1):
        for lang_ind, lang in enumerate(args.languages):
            ds.add_faiss_index(column=f'{lang}_embed_{i}')

    output = []
    for example in ds:
        output.append(copy.deepcopy(example))
        for lang_ind, lang in enumerate(args.languages):
            for i in range(1, 13, 1):
                scores, retrieved_examples = ds.get_nearest_examples(f'{lang}_embed_{i}',
                                                                     np.array(example[f'{lang}_embed_{i}'],
                                                                              dtype=np.float32), k=11)
                output[-1][f'{lang}_nearest_sentences_embed_{i}'] = [torch.tensor(e) for e in
                                                                     retrieved_examples[f'{lang}_embed_{i}'][1:]]

    return output


def load_embeddings(args, train=False):
    if train:
        file_name = f'multilingual_train_{args.num_of_train_embeddings}.pth'
    else:
        if args.faiss:
            file_name = f'multilingual_test_{args.num_of_test_embeddings}_faiss.pth'
        else:
            file_name = f'multilingual_test_{args.num_of_test_embeddings}.pth'
    if os.path.isfile(os.path.join(args.output_dir, file_name)):
        with open(os.path.join(args.output_dir, file_name), 'rb') as f:
            embedding_vectors = pickle.load(f)
    else:
        if train:
            embedding_vectors = create_dataset('train', device, args)
        else:
            # test
            if args.faiss:
                embedding_vectors = create_dataset_faiss('test', device, args)
            else:
                embedding_vectors = create_dataset('test', device, args)
        with open(os.path.join(args.output_dir, file_name), 'wb') as f:
            pickle.dump(embedding_vectors, f)

    if not args.faiss:
        num_embeddings = args.num_of_train_embeddings if train else args.num_of_test_embeddings
        embedding_vectors['tensor'] = embedding_vectors['tensor'][:, :num_embeddings, :, :]
    return embedding_vectors


def evaluate(args, embedding_vectors, trained_languages, encoder=None):
    embedding_vectors = embedding_vectors['tensor'].to(device)
    lang_indices = list(range(len(args.languages)))
    accuracies = []
    for lang1_ind in lang_indices:
        for lang2_ind in lang_indices:
            if lang1_ind == trained_languages[0] and lang2_ind == trained_languages[1] \
                    or lang1_ind == lang2_ind:
                continue
            pair_embeddings = embedding_vectors[[lang1_ind, lang2_ind]]  # [2, num_sentences, num_layers, out_dim]
            pair_acc = []
            for layer_num, layer in enumerate(pair_embeddings.swapaxes(0, 2)):
                num_correct = 0
                total = 0
                layer = layer.swapaxes(0, 1)
                # layer : 2 x num_sentence x out_dim
                for batch_id, (view_0, view_1) in enumerate(
                        zip(layer[0].split(args.split_size), layer[1].split(args.split_size))):
                    if view_0.shape[0] != args.split_size:
                        continue
                    # view_i : [args.split_size, out_dim]
                    accuracy_list = []
                    if args.sim_measure == 'CKA':
                        sim = feature_space_linear_cka(view_1.detach().cpu().numpy(),
                                                       view_0.detach().cpu().numpy(), debiased=args.debiased)
                        if np.isnan(sim):
                            raise Exception('Nan')
                    elif args.sim_measure == 'Dot':
                        sim = torch.dot(view_1.reshape(-1), view_0.reshape(-1))
                        if torch.isnan(sim):
                            raise Exception('Nan')
                    elif args.sim_measure == 'Norm':
                        sim = (1 - (F.normalize(view_1, dim=-1) -
                                    F.normalize(view_0, dim=-1)).norm(dim=-1, p=2)).mean()
                        if torch.isnan(sim):
                            raise Exception('Nan')
                    else:
                        sim = encoder(view_1, view_0)
                        if torch.isnan(sim):
                            raise Exception('Nan')
                    sim = sim.item()
                    accuracy_list.append(sim)

                    for _ in range(10):
                        temp_index = torch.randint(layer[0].shape[0], (view_0.shape[0],))
                        temporal_subset = layer[0][temp_index]
                        if args.sim_measure == 'CKA':
                            sim = feature_space_linear_cka(view_1.detach().cpu().numpy(),
                                                           temporal_subset.detach().cpu().numpy(),
                                                           debiased=args.debiased)
                        elif args.sim_measure == 'Dot':
                            sim = torch.dot(view_1.reshape(-1), temporal_subset.reshape(-1))
                        elif args.sim_measure == 'Norm':
                            sim = (1 - (F.normalize(view_1, dim=-1) -
                                        F.normalize(temporal_subset, dim=-1)).norm(dim=-1, p=2)).mean()
                        else:
                            sim = encoder(view_1, temporal_subset)
                            if torch.isnan(sim):
                                raise Exception('Nan')
                        sim = sim.item()
                        accuracy_list.append(sim)

                    accuracy_list = np.array(accuracy_list)
                    if accuracy_list.argmax() == 0:
                        num_correct += 1
                    else:
                        pass
                    total += 1

                print(
                    f'[{args.sim_measure}] Trained languages: {args.languages[trained_languages[0]]}, {args.languages[trained_languages[1]]}'
                    f' Pair: {args.languages[lang1_ind]}, {args.languages[lang2_ind]}. Layer: {layer_num}. Accuracy: {num_correct / total * 100:.3f}%',
                    flush=True)
                pair_acc.append(num_correct / total)

            accuracies.append(pair_acc)
            print()
    return np.array(accuracies)


def evaluate_faiss(args, embeddings, trained_languages, encoder=None):
    lang_indices = list(range(len(args.languages)))
    accuracies = []
    for lang1_ind in lang_indices:
        for lang2_ind in lang_indices:
            if lang1_ind == trained_languages[0] and lang2_ind == trained_languages[1] \
                    or lang1_ind == lang2_ind:
                continue
            lang1 = args.languages[lang1_ind]
            lang2 = args.languages[lang2_ind]
            pair_acc = []
            for layer_num in range(1, 13, 1):
                num_correct = 0
                total = 0
                for batch in [embeddings[i:i + args.split_size] for i in range(0, len(embeddings), args.split_size)]:
                    if len(batch) != args.split_size:
                        continue
                    accuracy_list = []
                    if args.sim_measure == 'CKA':
                        true_sentence = np.stack([np.array(b[f'{lang2}_embed_{layer_num}']) for b in batch])
                        sim = feature_space_linear_cka(true_sentence, np.stack(
                            [np.array(b[f'{lang1}_embed_{layer_num}']) for b in batch]),
                                                       debiased=args.debiased)
                        if np.isnan(sim):
                            raise Exception('Nan')
                    elif args.sim_measure == 'Dot':
                        true_sentence = torch.stack([torch.tensor(b[f'{lang2}_embed_{layer_num}']) for b in batch]).to(
                            device)
                        sim = torch.dot(true_sentence.reshape(-1),
                                        torch.stack([torch.tensor(b[f'{lang1}_embed_{layer_num}']) for b in batch]).to(
                                            device).reshape(-1))
                        if torch.isnan(sim):
                            raise Exception('Nan')
                    elif args.sim_measure == 'Norm':
                        true_sentence = torch.stack([torch.tensor(b[f'{lang2}_embed_{layer_num}']) for b in batch]).to(
                            device)
                        sim = (1 - (F.normalize(true_sentence, dim=-1) -
                                    F.normalize(
                                        torch.stack([torch.tensor(b[f'{lang1}_embed_{layer_num}']) for b in batch]).to(
                                            device), dim=-1)).norm(dim=-1, p=2)).mean()
                        if torch.isnan(sim):
                            raise Exception('Nan')
                    else:
                        true_sentence = torch.stack([torch.tensor(b[f'{lang2}_embed_{layer_num}']) for b in batch]).to(
                            device)
                        sim = encoder(true_sentence,
                                      torch.stack([torch.tensor(b[f'{lang1}_embed_{layer_num}']) for b in batch]).to(
                                          device))
                        if torch.isnan(sim):
                            raise Exception('Nan')
                    sim = sim.item()
                    accuracy_list.append(sim)

                    for j in range(10):
                        if args.sim_measure == 'CKA':
                            current_embed = np.stack(
                                [b[f'{lang1}_nearest_sentences_embed_{layer_num}'][j].detach().cpu().numpy() for b in
                                 batch])
                            sim = feature_space_linear_cka(true_sentence, current_embed, debiased=args.debiased)
                        elif args.sim_measure == 'Dot':
                            current_embed = torch.stack(
                                [b[f'{lang1}_nearest_sentences_embed_{layer_num}'][j] for b in batch]).to(device)
                            sim = torch.dot(true_sentence.reshape(-1), current_embed.reshape(-1))
                            if torch.isnan(sim):
                                raise Exception('Nan')

                        elif args.sim_measure == 'Norm':
                            current_embed = torch.stack(
                                [b[f'{lang1}_nearest_sentences_embed_{layer_num}'][j] for b in batch]).to(device)
                            sim = (1 - (F.normalize(true_sentence, dim=-1) -
                                        F.normalize(current_embed, dim=-1)).norm(dim=-1, p=2)).mean()
                            if torch.isnan(sim):
                                raise Exception('Nan')
                        else:
                            current_embed = torch.stack(
                                [b[f'{lang1}_nearest_sentences_embed_{layer_num}'][j] for b in batch]).to(device)
                            sim = encoder(true_sentence, current_embed)
                            if torch.isnan(sim):
                                raise Exception('Nan')
                        sim = sim.item()
                        accuracy_list.append(sim)

                    accuracy_list = np.array(accuracy_list)
                    if accuracy_list.argmax() == 0:
                        num_correct += 1
                    total += 1

                print(
                    f'[{args.sim_measure} FAISS] Trained languages: {args.languages[trained_languages[0]]}, {args.languages[trained_languages[1]]}'
                    f' Pair: {lang1}, {lang2}. Layer: {layer_num}. Accuracy: {num_correct / total * 100:.3f}%',
                    flush=True)
                pair_acc.append(num_correct / total)

            accuracies.append(pair_acc)
            print()
    return np.array(accuracies)


def get_encoder(args, train_embedding_vectors, encoder, encoder_name, trained_languages):
    train_embedding_vectors = train_embedding_vectors[
        [trained_languages[0], trained_languages[1]]].to(
        device)  # [num_languages, num_sentences, num_layers, out_dim]
    if os.path.isfile(os.path.join(args.output_dir, encoder_name)):
        checkpoint = torch.load(os.path.join(args.output_dir, encoder_name), map_location=device)
        encoder.load_state_dict(checkpoint)
    else:
        # Train it
        optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.1)
        for epoch in range(args.epochs):
            epoch_loss = 0
            num_examples = 0

            for layer_num, layer in enumerate(
                    train_embedding_vectors.swapaxes(0, 2)[torch.randperm(train_embedding_vectors.shape[2])]):
                layer = layer.swapaxes(0, 1)
                # layer : 2 x num_sentence x out_dim
                for batch_id, (view_0, view_1) in enumerate(
                        zip(layer[0].split(args.train_batch_size), layer[1].split(args.train_batch_size))):
                    if view_0.shape[0] != args.train_batch_size:
                        continue
                    accuracy = encoder(view_0, view_1)
                    loss = -accuracy
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += loss.item()
                    num_examples += 1

            print(f'[{args.sim_measure}] epoch: {epoch} loss: {epoch_loss / num_examples}')

        torch.save(encoder.state_dict(), os.path.join(args.output_dir, encoder_name))

    return encoder


def multilingual_benchmark_deep(args, device):
    if not os.path.isdir(os.path.join(args.output_dir, args.models_dir)):
        os.mkdir(os.path.join(args.output_dir, args.models_dir))

    train_embedding_vectors = load_embeddings(args, train=True)
    train_embedding_vectors = train_embedding_vectors['tensor'].to(device)
    lang_indices = list(range(len(args.languages)))
    embedding_vectors = load_embeddings(args)
    accuracies = []
    for lang1_ind in lang_indices:
        for lang2_ind in lang_indices:
            if lang1_ind == lang2_ind:
                continue
            if args.sim_measure == 'DeepCKA':
                encoder_name = f'{args.models_dir}/encoder_deep_cka_{args.languages[lang1_ind]}_{args.languages[lang2_ind]}.pth'
                encoder = deepCKA(train_embedding_vectors.shape[3], args.out_dim, mid_layers=args.mid_layers).to(device)
            else:
                # DeepDot
                encoder_name = f'{args.models_dir}/encoder_deep_dot_{args.languages[lang1_ind]}_{args.languages[lang2_ind]}.pth'
                encoder = deepDot(train_embedding_vectors.shape[3], args.out_dim, mid_layers=args.mid_layers).to(device)
            encoder = get_encoder(args, train_embedding_vectors, encoder, encoder_name, (lang1_ind, lang2_ind))
            if args.faiss:
                encoder_acc = evaluate_faiss(args, embedding_vectors, (lang1_ind, lang2_ind), encoder)
            else:
                encoder_acc = evaluate(args, embedding_vectors, (lang1_ind, lang2_ind), encoder)
            accuracies.append(encoder_acc)

    accuracies = np.array(accuracies)  # [20 x 19 x 12]
    faiss = ''
    if args.faiss:
        faiss = '_faiss'
    accuracies_reshaped = accuracies.reshape(-1, accuracies.shape[-1]) * 100
    print(f'[{args.sim_meassure}{faiss}] Mean accuracy: {accuracies_reshaped.mean(axis=0)}')
    print(f'[{args.sim_meassure}{faiss}] Std: {accuracies_reshaped.std(axis=0)}')


def multilingual_benchmark_contrastive(args, device):
    if not os.path.isdir(os.path.join(args.output_dir, args.models_dir)):
        os.mkdir(os.path.join(args.output_dir, args.models_dir))

    embedding_vectors = load_embeddings(args)
    # sentences = embedding_vectors['sentences']
    # indices = embedding_vectors['indices']

    train_embedding_vectors = load_embeddings(args, train=True)
    train_embedding_vectors = train_embedding_vectors['tensor'].to(device)
    # train_sentences = train_embedding_vectors['sentences']
    lang_indices = list(range(len(args.languages)))

    accuracies = []
    for lang1_ind in lang_indices:
        for lang2_ind in lang_indices:
            if lang1_ind == lang2_ind:
                continue
            if args.sim_measure == 'contrastive':
                encoder_name = f'{args.models_dir}/encoder_contrastive_{args.languages[lang1_ind]}_{args.languages[lang2_ind]}.pth'
                encoder = ContrastiveSim(train_embedding_vectors.shape[3], args.out_dim, mid_layers=args.mid_layers).to(
                    device)
            else:
                # contrastive_dis
                encoder_name = f'{args.models_dir}/encoder_contrastive_dis_{args.languages[lang1_ind]}_{args.languages[lang2_ind]}.pth'
                encoder = ContrastiveSim_dis(train_embedding_vectors.shape[3], args.out_dim,
                                             mid_layers=args.mid_layers).to(device)
            if os.path.isfile(os.path.join(args.output_dir, encoder_name)):
                checkpoint = torch.load(os.path.join(args.output_dir, encoder_name), map_location=device)
                encoder.load_state_dict(checkpoint)
            else:
                # Train it
                optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
                if args.sim_measure == 'contrastive':
                    criterion = SupConLoss(temperature=0.07).to(device)
                else:
                    criterion = SupConLossDis(temperature=0.07).to(device)
                for epoch in range(args.epochs):
                    epoch_loss = 0
                    num_examples = 0

                    for batch_id, batch in enumerate(
                            train_embedding_vectors[[lang1_ind, lang2_ind]].split(args.train_batch_size, dim=1)):
                        if batch.shape[1] != args.train_batch_size:
                            continue
                        for i, layer in enumerate(
                                batch.swapaxes(0, 2)[torch.randperm(batch.shape[2])]):
                            layer = layer.swapaxes(0, 1)
                            # layer : num_languages x num_sentence x out_dim

                            encoded_features = torch.empty(
                                (batch.shape[0], batch.shape[1], args.out_dim)).to(
                                device)  # [2 x num_examples x out_dim]
                            for j, view in enumerate(layer):
                                # view: num_samples x out_dim
                                encoded_features[j, :, :] = encoder.get_features(view)

                            encoded_features = encoded_features.swapaxes(0, 1)
                            # encoded_features: [num_examples, 2, out_dim]
                            loss = criterion(encoded_features)
                            epoch_loss += loss.item()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            num_examples += 1

                    print(f'[{args.sim_measure}] epoch: {epoch} loss: {epoch_loss / num_examples}')

                torch.save(encoder.state_dict(), os.path.join(args.output_dir, encoder_name))

            encoder_acc = evaluate(args, embedding_vectors, (lang1_ind, lang2_ind), encoder)
            accuracies.append(encoder_acc)

    accuracies = np.array(accuracies)
    accuracies_reshaped = accuracies.reshape(-1, accuracies.shape[-1]) * 100
    print(f'[{args.sim_measure}] Mean accuracy: {accuracies_reshaped.mean(axis=0)}')
    print(f'[{args.sim_measure}] Std: {accuracies_reshaped.std(axis=0)}')

def multilingual_benchmark_closed_form(args, device):
    if not os.path.isdir(os.path.join(args.output_dir, args.models_dir)):
        os.mkdir(os.path.join(args.output_dir, args.models_dir))

    # lang_indices = list(range(len(args.languages)))
    embedding_vectors = load_embeddings(args)
    accuracies = []

    if args.faiss:
        encoder_acc = evaluate_faiss(args, embedding_vectors, (0, 0))
    else:
        # sentences = embedding_vectors['sentences']
        # indices = embedding_vectors['indices']
        # embedding_vectors = embedding_vectors['tensor'].to(device)  # [num_languages, num_sentences, num_layers, out_dim]
        encoder_acc = evaluate(args, embedding_vectors, (0, 0))
    accuracies.append(encoder_acc)

    accuracies = np.array(accuracies)  # [20 x 19 x 12]
    faiss = ''
    if args.faiss:
        faiss = '_faiss'

    accuracies_reshaped = accuracies.reshape(-1, accuracies.shape[-1]) * 100
    print(f'[{args.sim_measure}{faiss}] Mean accuracy: {accuracies_reshaped.mean(axis=0)}')
    print(f'[{args.sim_measure}{faiss}] Std: {accuracies_reshaped.std(axis=0)}')


def multilingual_benchmark_contrastive_faiss(args, device):
    if not os.path.isdir(os.path.join(args.output_dir, args.models_dir)):
        os.mkdir(os.path.join(args.output_dir, args.models_dir))

    # FAISS dataset for evaluation
    embeddings = load_embeddings(args)

    # Regular dataset for model training
    train_embedding_vectors = load_embeddings(args, train=True)
    train_embedding_vectors = train_embedding_vectors['tensor'].to(device)
    lang_indices = list(range(len(args.languages)))

    accuracies = []
    for lang1_ind in lang_indices:
        for lang2_ind in lang_indices:
            if lang1_ind == lang2_ind:
                continue
            if args.sim_measure == 'contrastive':
                encoder_name = f'{args.models_dir}/encoder_contrastive_{args.languages[lang1_ind]}_{args.languages[lang2_ind]}.pth'
                encoder = ContrastiveSim(train_embedding_vectors.shape[3], args.out_dim, mid_layers=args.mid_layers).to(
                    device)
            else:
                # contrastive_dis
                encoder_name = f'{args.models_dir}/encoder_contrastive_dis_{args.languages[lang1_ind]}_{args.languages[lang2_ind]}.pth'
                encoder = ContrastiveSim_dis(train_embedding_vectors.shape[3], args.out_dim,
                                             mid_layers=args.mid_layers).to(
                    device)
            if os.path.isfile(os.path.join(args.output_dir, encoder_name)):
                checkpoint = torch.load(os.path.join(args.output_dir, encoder_name), map_location=device)
                encoder.load_state_dict(checkpoint)
            else:
                # Train it
                optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
                if args.sim_measure == 'cintrastive':
                    criterion = SupConLoss(temperature=0.07).to(device)
                else:
                    criterion = SupConLossDis(temperature=0.07).to(device)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100)
                for epoch in range(args.epochs):
                    epoch_loss = 0
                    num_examples = 0

                    for batch_id, batch in enumerate(
                            train_embedding_vectors[[lang1_ind, lang2_ind]].split(args.train_batch_size, dim=1)):
                        if batch.shape[1] != args.train_batch_size:
                            continue
                        for i, layer in enumerate(
                                batch.swapaxes(0, 2)[torch.randperm(batch.shape[2])]):
                            layer = layer.swapaxes(0, 1)
                            # layer : num_languages x num_sentence x out_dim

                            encoded_features = torch.empty(
                                (batch.shape[0], batch.shape[1], args.out_dim)).to(
                                device)  # [2 x num_examples x out_dim]
                            for j, view in enumerate(layer):
                                # view: num_samples x out_dim
                                encoded_features[j, :, :] = encoder.get_features(view)

                            encoded_features = encoded_features.swapaxes(0, 1)
                            # encoded_features: [num_examples, 2, out_dim]
                            loss = criterion(encoded_features)
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            epoch_loss += loss.item()
                            num_examples += 1

                    print(f'[{args.sim_measure}] epoch: {epoch} loss: {epoch_loss / num_examples}')

                torch.save(encoder.state_dict(), os.path.join(args.output_dir, encoder_name))

            encoder_acc = evaluate_faiss(args, embeddings, (lang1_ind, lang2_ind), encoder)
            accuracies.append(encoder_acc)

    accuracies = np.array(accuracies)
    accuracies_reshaped = accuracies.reshape(-1, accuracies.shape[-1]) * 100
    print(f'[{args.sim_measure} FAISS] Mean accuracy: {accuracies_reshaped.mean(axis=0)}')
    print(f'[{args.sim_measure} FAISS] Std: {accuracies_reshaped.std(axis=0)}')

def get_args():
    parser = argparse.ArgumentParser(description='Parameters for layer prediction')
    parser.add_argument('--output-dir', type=str, default='multilingual_benchmark')
    parser.add_argument('--models-dir', type=str, default='models')
    parser.add_argument('--languages', type=list, default=['en', 'ar', 'zh', 'ru', 'tr'])
    parser.add_argument('--num-of-test-embeddings', type=int, default=5000)
    parser.add_argument('--num-of-train-embeddings', type=int, default=10000)
    parser.add_argument('--split-size', type=int, default=8)
    parser.add_argument('--train-batch-size', type=int, default=1024)

    parser.add_argument('--sim-measure', type=str, default='CKA', choices=['CKA', 'DeepDot', 'DeepCKA', 'contrastive',
                                                                           'contrastive_dis', 'Norm', 'Dot'])
    parser.add_argument('--faiss', action='store_true', default=True)
    parser.add_argument('--do-all', action='store_true', default=False)
    parser.add_argument('--debiased', action='store_true', default=True)

    # Encoder configuration
    parser.add_argument('--out-dim', type=int, default=128)
    parser.add_argument('--mid-layers', type=list, default=[512, 256])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.manual_seed(42)
    transformers.logging.set_verbosity_error()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device->{device}')
    args = get_args()
    print(args)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if args.do_all or args.sim_measure == 'CKA':
        args.sim_measure = 'CKA'
        multilingual_benchmark_closed_form(args, device)

    if args.do_all or args.sim_measure == 'Dot':
        args.sim_measure = 'Dot'
        multilingual_benchmark_closed_form(args, device)

    if args.do_all or args.sim_measure == 'Norm':
        args.sim_measure = 'Norm'
        multilingual_benchmark_closed_form(args, device)

    if args.do_all or args.sim_measure == 'DeepCKA':
        args.sim_measure = 'DeepCKA'
        multilingual_benchmark_deep(args, device)

    if args.do_all or args.sim_measure == 'DeepDot':
        args.sim_measure = 'DeepDot'
        multilingual_benchmark_deep(args, device)

    if args.do_all or args.sim_measure == 'contrastive':
        args.sim_measure = 'contrastive'
        if args.faiss:
            multilingual_benchmark_contrastive_faiss(args, device)
        else:
            multilingual_benchmark_contrastive(args, device)

    if args.do_all or args.sim_measure == 'contrastive_dis':
        args.sim_measure = 'contrastive_dis'
        if args.faiss:
            multilingual_benchmark_contrastive_faiss(args, device)
        else:
            multilingual_benchmark_contrastive(args, device)

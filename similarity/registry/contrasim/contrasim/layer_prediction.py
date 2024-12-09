from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from pwcca import compute_pwcca
from cca_core import compute_svcca
from cka import feature_space_linear_cka
from deep_sim import deepDot, deepCKA, ContrastiveSim, ContrastiveSim_dis
from utils import SupConLoss, SupConLossDis
import torch.nn.functional as F
import os
import pickle
import numpy as np
import torch
import argparse
import transformers
import datasets


def get_hidden_states(encoded, token_ids_word, model, layers):
    output = model(**encoded)
    states = output.hidden_states
    layers = layers or list(range(1, len(output.hidden_states)))
    output = torch.stack([states[i].detach().squeeze()[token_ids_word] for i in layers])
    word_tokens_output = output.mean(axis=1).detach()
    return word_tokens_output.squeeze().cpu().numpy()


def get_word_vector(sent, tokenizer, model, layers, device):
    encoded = tokenizer(sent, return_tensors="pt")
    for key in encoded:
        encoded[key] = encoded[key].to(device)
    hidden_states = {}
    for word_id in set(encoded.word_ids()):
        if word_id is None:
            continue
        token_ids_word = np.where(np.array(encoded.word_ids()) == word_id)
        hidden_states[''.join(np.asarray(encoded.tokens())[token_ids_word]).replace('#', '').replace('\u0120', '')] \
            = get_hidden_states(encoded,
                                token_ids_word,
                                model,
                                layers)
    return hidden_states


def create_dataset(split, model_names, args, device):
    models = {}
    tokenizers = {}
    with torch.no_grad():
        for model_name in model_names:
            print('Loading model: ', model_name)
            models[model_name] = AutoModel.from_pretrained(model_name,
                                                           output_hidden_states=True,
                                                           output_attentions=True).to(device)
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
    if args.dataset == 'ptb_text_only':
        dataset = load_dataset(args.dataset, split=split)
        sentences = dataset['sentence']
    elif args.dataset == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
        sentences = dataset['text']
    else:
        print(f'Dataset: {args.dataset} not supported')
        exit(1)

    word_embeddings = {}
    for model_name in model_names:
        word_embeddings[model_name] = {}
        word_embeddings[model_name]['words'] = []
        word_embeddings[model_name]['embeddings'] = []

    num_embeddings = args.num_of_test_embeddings if split == 'test' else args.num_of_train_embeddings
    for model_name in model_names:
        print(f'Current model: {model_name}')
        i = -1
        while(len(word_embeddings[model_name]['embeddings']) < num_embeddings):
            i += 1
            if len(sentences[i]) == 0 or '=' in sentences[i]:
                continue
            embedding_dict = get_word_vector(sentences[i], tokenizers[model_name], models[model_name],
                                             layers=None, device=device)
            words, embeddings = zip(*embedding_dict.items())
            word_embeddings[model_name]['words'].extend(words)
            word_embeddings[model_name]['embeddings'].extend(embeddings)

    embedding_vectors = []
    for model_name in model_names:
        em_array = np.array([word_embeddings[model_name]['embeddings']]).squeeze(0).transpose(1, 0, 2)
        embedding_vectors.append(em_array)

    return np.stack(embedding_vectors)


def load_embeddings(args, model_names, train=False):
    output_dir = args.output_dir
    if train:
        file_name = f'train_embedding_vectors_{args.dataset}_{args.num_of_train_embeddings}.pth'
    else:
        file_name = f'embedding_vectors_{args.dataset}_{args.num_of_test_embeddings}.pth'

    if os.path.isfile(os.path.join(output_dir, file_name)):
        with open(os.path.join(output_dir, file_name), 'rb') as f:
            embedding_vectors = pickle.load(f)
    else:
        embedding_vectors = create_dataset('train' if train else 'test', model_names, args, device)
        with open(os.path.join(args.output_dir, file_name),
                  'wb') as f:
            pickle.dump(embedding_vectors, f)

    return embedding_vectors


def evaluate(args, model_names, embedding_vectors, encoder=None, print_result=True):
    benchmark_accuracy = 0
    for pair in zip(*[iter([i for i in range(len(model_names))])] * 2):
        model1_num, model2_num = pair
        total = 0
        num_correct = 0
        for layer_num, layer in enumerate(embedding_vectors[model1_num]):
            similarity_list = []
            for _, second_layer in enumerate(embedding_vectors[model2_num]):
                if args.sim_measure == 'CKA':
                    similarity = feature_space_linear_cka(layer, second_layer)
                    if np.isnan(similarity):
                        raise Exception('Nan')
                elif args.sim_measure == 'CCA':
                    similarity = compute_pwcca(layer, second_layer)
                    if np.isnan(similarity):
                        raise Exception('Nan')

                elif args.sim_measure == 'svcca':
                    similarity = compute_svcca(layer, second_layer)
                    if np.isnan(similarity):
                        raise Exception('Nan')

                elif args.sim_measure == 'Norm':
                    similarity = (1 - (F.normalize(torch.from_numpy(layer).to(device), dim=-1) -
                                F.normalize(torch.from_numpy(second_layer).to(device), dim=-1)).norm(dim=-1, p=2)).mean()

                elif args.sim_measure == 'Dot':
                    similarity = torch.dot(torch.from_numpy(layer).to(device).reshape(-1), torch.from_numpy(second_layer).to(device).reshape(-1))

                else:
                    similarity = encoder(torch.from_numpy(layer).to(device), torch.from_numpy(second_layer).to(device))
                    if torch.isnan(similarity):
                        raise Exception('Nan')
                similarity = similarity.item()
                similarity_list.append(similarity)

            similarity_list = np.array(similarity_list)
            if similarity_list.argmax() == layer_num:
                num_correct += 1
            else:
                pass
            total += 1

        benchmark_accuracy += num_correct / total
    if print_result:
        print(f'[{args.sim_measure}] Accuracy: {benchmark_accuracy / (len(model_names)/2) * 100:.3f}%')  # /2 because we take pairs of models

    return benchmark_accuracy / (len(model_names)/2)


def get_encoder(args, model_names, train_embedding_vectors, encoder, encoder_name):
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
            for pair in zip(*[iter([i for i in range(len(model_names))])] * 2):
                model1_num, model2_num = pair[0], pair[1]
                perm = torch.randperm(train_embedding_vectors.shape[0])
                for layer_num, layer in enumerate(train_embedding_vectors[model1_num][perm]):
                    for second_layer_num, second_layer in enumerate(train_embedding_vectors[model2_num][perm]):
                        similarity = encoder(torch.from_numpy(layer).to(device), torch.from_numpy(second_layer).to(device))
                        loss = -similarity
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        epoch_loss += loss.item()
                        num_examples += 1

            print(f'[{args.sim_measure}] epoch: {epoch} loss: {epoch_loss / num_examples}')

        torch.save(encoder.state_dict(), os.path.join(args.output_dir, encoder_name))

    return encoder


def layer_prediction_cca(args):
    model_names = [f'google/multiberts-seed_{i}' for i in range(args.num_models)]
    embedding_vectors = load_embeddings(args, model_names)
    evaluate(args, model_names, embedding_vectors)


def layer_prediction_cka(args):
    model_names = [f'google/multiberts-seed_{i}' for i in range(args.num_models)]
    embedding_vectors = load_embeddings(args, model_names)
    evaluate(args, model_names, embedding_vectors, model_names)


def layer_prediction_deep_dot(args):
    if not os.path.isdir(os.path.join(args.output_dir, 'models')):
        os.mkdir(os.path.join(args.output_dir, 'models'))
    model_names = [f'google/multiberts-seed_{i}' for i in range(args.num_models)]

    embedding_vectors = load_embeddings(args, model_names)
    train_embedding_vectors = load_embeddings(args, model_names, train=True)

    encoder = deepDot(embedding_vectors.shape[3], args.out_dim, mid_layers=args.mid_layers).to(device)
    encoder_name = f'models/encoder_deep_dot_{args.dataset}.pth'
    encoder = get_encoder(args, model_names, train_embedding_vectors, encoder, encoder_name)

    evaluate(args, model_names, embedding_vectors, encoder)


def layer_prediction_deep_cka(args):
    if not os.path.isdir(os.path.join(args.output_dir, 'models')):
        os.mkdir(os.path.join(args.output_dir, 'models'))
    model_names = [f'google/multiberts-seed_{i}' for i in range(args.num_models)]

    embedding_vectors = load_embeddings(args, model_names)
    train_embedding_vectors = load_embeddings(args, model_names, train=True)

    encoder = deepCKA(embedding_vectors.shape[3], args.out_dim, mid_layers=args.mid_layers).to(device)
    encoder_name = f'models/encoder_deep_cka_{args.dataset}.pth'
    encoder = get_encoder(args, model_names, train_embedding_vectors, encoder, encoder_name)
    evaluate(args, model_names, embedding_vectors, encoder)


def layer_prediction_contrastive(args):
    if not os.path.isdir(os.path.join(args.output_dir, 'models')):
        os.mkdir(os.path.join(args.output_dir, 'models'))
    model_names = [f'google/multiberts-seed_{i}' for i in range(args.num_models)]

    embedding_vectors = load_embeddings(args, model_names)
    train_embedding_vectors = load_embeddings(args, model_names, train=True)

    encoder = ContrastiveSim(embedding_vectors.shape[3], args.out_dim, mid_layers=args.mid_layers).to(device)
    if os.path.isfile(os.path.join(args.output_dir, f'models/encoder_contrastive_{args.dataset}.pth')):
        checkpoint = torch.load(os.path.join(args.output_dir, f'models/encoder_contrastive_{args.dataset}.pth'), map_location=device)
        encoder.load_state_dict(checkpoint)
    else:
        # Train it
        optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.1)
        criterion = SupConLoss(temperature=0.07).to(device)

        for epoch in range(args.epochs):
            epoch_loss = 0
            num_batches = 0
            for batch in torch.from_numpy(train_embedding_vectors).split(args.train_batch_size, dim=2):
                if torch.isnan(batch.max()):
                    raise Exception('Nan')
                encoded_features = torch.empty((batch.shape[1], batch.shape[0], batch.shape[2],
                                                args.out_dim))  # [num_layers x num_models x num_examples x out_dim]
                for j, model in enumerate(batch):
                    for i, model_layer in enumerate(model):
                        encoder_out = encoder.get_features(model_layer.to(device))
                        encoded_features[i, j, :, :] = encoder_out / torch.norm(encoder_out)

                loss = criterion(encoded_features)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            scheduler.step()
            print(f'[Contrastive] epoch: {epoch} loss: {epoch_loss}')

        torch.save(encoder.state_dict(), os.path.join(args.output_dir, f'models/encoder_contrastive_{args.dataset}.pth'))

    evaluate(args, model_names, embedding_vectors, encoder)


def layer_prediction_contrastive_dis(args):
    if not os.path.isdir(os.path.join(args.output_dir, 'models')):
        os.mkdir(os.path.join(args.output_dir, 'models'))
    model_names = [f'google/multiberts-seed_{i}' for i in range(args.num_models)]

    embedding_vectors = load_embeddings(args, model_names)
    train_embedding_vectors = load_embeddings(args, model_names, train=True)

    encoder = ContrastiveSim_dis(embedding_vectors.shape[3], args.out_dim, mid_layers=args.mid_layers).to(device)
    if os.path.isfile(os.path.join(args.output_dir, f'models/encoder_contrastive_dis_{args.dataset}.pth')):
        checkpoint = torch.load(os.path.join(args.output_dir, f'models/encoder_contrastive_dis_{args.dataset}.pth'), map_location=device)
        encoder.load_state_dict(checkpoint)
    else:
        # Train it
        optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.1)
        criterion = SupConLossDis(temperature=0.07).to(device)

        for epoch in range(args.epochs):
            epoch_loss = 0
            num_batches = 0
            for batch in torch.from_numpy(train_embedding_vectors).split(args.train_batch_size, dim=2):
                if torch.isnan(batch.max()):
                    raise Exception('Nan')
                encoded_features = torch.empty((batch.shape[1], batch.shape[0], batch.shape[2],
                                                args.out_dim))  # [num_layers x num_models x num_examples x out_dim]
                for j, model in enumerate(batch):
                    for i, model_layer in enumerate(model):
                        encoder_out = encoder.get_features(model_layer.to(device))
                        encoded_features[i, j, :, :] = encoder_out / torch.norm(encoder_out)

                loss = criterion(encoded_features)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            scheduler.step()
            print(f'[Contrastive] epoch: {epoch} loss: {epoch_loss}')

        torch.save(encoder.state_dict(), os.path.join(args.output_dir, f'models/encoder_contrastive_dis_{args.dataset}.pth'))

    evaluate(args, model_names, embedding_vectors, encoder)


def get_args():
    parser = argparse.ArgumentParser(description='Parameters for layer prediction')
    parser.add_argument('--output-dir', type=str, default='layer_prediction')
    parser.add_argument('--dataset', type=str, default='ptb_text_only', choices=['ptb_text_only', 'wikitext'])
    parser.add_argument('--num-of-test-embeddings', type=int, default=5000)
    parser.add_argument('--num-of-train-embeddings', type=int, default=10000)
    parser.add_argument('--num-models', type=int, default=10)
    parser.add_argument('--sim-measure', type=str, default='both', choices=['CCA', 'CKA', 'DeepDot',
                                                                            'DeepCKA', 'contrastive', 'svcca',
                                                                            'contrastive_dis',  'Dot', 'Norm'])
    parser.add_argument('--do-all', action='store_true', default=False)


    # Encoder configuration
    parser.add_argument('--out-dim', type=int, default=128)
    parser.add_argument('--train-batch-size', type=int, default=1024)
    parser.add_argument('--mid-layers', type=list, default=[512, 256])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device->{device}')
    args = get_args()
    print(args)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if args.do_all or args.sim_measure == 'CCA':
        args.sim_measure = 'CCA'
        layer_prediction_cca(args)

    if args.do_all or args.sim_measure == 'svcca':
        args.sim_measure = 'svcca'
        layer_prediction_cca(args)

    if args.do_all or args.sim_measure == 'Dot':
        args.sim_measure = 'Dot'
        layer_prediction_cca(args)

    if args.do_all or args.sim_measure == 'Norm':
        args.sim_measure = 'Norm'
        layer_prediction_cca(args)

    if args.do_all or args.sim_measure == 'CKA':
        args.sim_measure = 'CKA'
        layer_prediction_cka(args)

    if args.do_all or args.sim_measure == 'DeepDot':
        args.sim_measure = 'DeepDot'
        layer_prediction_deep_dot(args)

    if args.do_all or args.sim_measure == 'DeepCKA':
        args.sim_measure = 'DeepCKA'
        layer_prediction_deep_cka(args)

    if args.do_all or args.sim_measure == 'contrastive':
        args.sim_measure = 'contrastive'
        layer_prediction_contrastive(args)

    if args.do_all or args.sim_measure == 'contrastive_dis':
        args.sim_measure = 'contrastive_dis'
        layer_prediction_contrastive_dis(args)

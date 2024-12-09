import argparse
import os
import pickle

import datasets
import numpy as np
import requests
import torch
import torch.nn.functional as F
from deep_sim import deepDot, deepCKA, ContrastiveSim, ContrastiveSim_dis
import transformers
from PIL import Image
from datasets import load_dataset
from cka import feature_space_linear_cka
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
from utils import SupConLoss, SupConLossDis
import signal


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def prepare_dataset(output_dir, device, args):
    with torch.no_grad():
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.vision_model)
        image_model = AutoModel.from_pretrained(args.vision_model).to(device)
        text_model = AutoModel.from_pretrained(args.text_model, output_hidden_states=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        if args.text_model == 'gpt2':
            padding = False
            token_ind = -1
        elif args.text_model == 'bert-base-cased':
            padding = True
            token_ind = 0
        else:
            raise Exception('Unknown text model')

        def get_representations(image, caption):
            inputs = feature_extractor(images=image, return_tensors="pt").to(device)
            outputs = image_model(**inputs)
            if 'vit' in args.vision_model:
                image_features = outputs.last_hidden_state.squeeze()[0]
            elif 'convnext' in args.vision_model:
                image_features = outputs.last_hidden_state.mean([-2, -1]).squeeze()
            else:
                pass

            inputs = tokenizer(caption, padding=padding, return_tensors="pt").to(device)
            outputs = text_model(**inputs)
            text_features = outputs.last_hidden_state.squeeze()[token_ind]
            d = {'image_f': image_features.detach().cpu(), 'image_url': pair['image_url'],
                 'caption_f': text_features.detach().cpu(),
                 'caption': caption}
            return d

        output = []
        dataset = load_dataset("conceptual_captions", split='train')
        done_example = 0
        for i, pair in enumerate(dataset):
            if i % 10 == 0:
                print(f'{i}/{len(dataset)}', flush=True)
            image = pair['image_url']
            caption = pair['caption']
            signal.alarm(15)
            try:
                image = Image.open(requests.get(image, stream=True).raw)
            except:
                print(f'Skipping example {i}', flush=True)
                continue
            else:
                # Reset the alarm
                signal.alarm(0)
            if image.mode != 'RGB':
                print(f'Skipping example {i}', flush=True)
                continue
            signal.alarm(15)
            try:
                d = get_representations(image, caption)
                output.append(d)
                done_example += 1
                if done_example == args.num_of_train_embeddings:
                    break
            except TimeoutException:
                print(f'Skipping example {i}', flush=True)
                continue
            else:
                signal.alarm(0)

        print(f'Number of train examples: {len(output)}')
        with open(os.path.join(output_dir, f'train_features_dict.pth'), 'wb') as f:
            pickle.dump(output, f)

        examples_torch = torch.stack([torch.stack((e['image_f'], e['caption_f'])).squeeze(1) for e in output])

        with open(os.path.join(output_dir, f'train_features_torch.pth'), 'wb') as f:
            pickle.dump(examples_torch, f)

        output = []
        dataset = load_dataset("conceptual_captions", split='validation')
        done_example = 0
        for i, pair in enumerate(dataset):
            if i % 10 == 0:
                print(f'{i}/{len(dataset)}', flush=True)
            image = pair['image_url']
            caption = pair['caption']
            signal.alarm(15)
            try:
                image = Image.open(requests.get(image, stream=True).raw)
            except:
                print(f'Skipping example {i}', flush=True)
                continue
            else:
                signal.alarm(0)
            if image.mode != 'RGB':
                print(f'Skipping example {i}', flush=True)
                continue
            signal.alarm(15)
            try:
                d = get_representations(image, caption)
                output.append(d)
                done_example += 1
                if done_example == args.num_of_test_embeddings:
                    break
            except TimeoutException:
                print(f'Skipping example {i}', flush=True)
                continue
            else:
                signal.alarm(0)

        print(f'Number of validation examples: {len(output)}')
        with open(os.path.join(output_dir, 'validation_features_dict.pth'), 'wb') as f:
            pickle.dump(output, f)

        examples_torch = torch.stack([torch.stack((e['image_f'], e['caption_f'])).squeeze(1) for e in output])

        with open(os.path.join(output_dir, 'validation_features_torch.pth'), 'wb') as f:
            pickle.dump(examples_torch, f)


def nearest_examples_dataset(args, device):
    with torch.no_grad():
        with open(os.path.join(args.output_dir, f'validation_features_dict.pth'), 'rb') as f:
            embeddings = pickle.load(f)

        text_model = AutoModel.from_pretrained(args.text_model).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        dataset = load_dataset("conceptual_captions", split='train')

        if args.text_model == 'gpt2':
            padding = False
            token_ind = -1
        elif args.text_model == 'bert-base-cased':
            padding = True
            token_ind = 0
        def get_text_embeddings(example):
            caption = example['caption']
            if caption in seen_captions:
                return np.full((1, 768), np.nan, dtype=np.float32)
            seen_captions.append(caption)
            inputs = tokenizer(caption, padding=padding, return_tensors="pt").to(device)
            try:
                outputs = text_model(**inputs)
                embeddings = outputs.last_hidden_state.squeeze()[token_ind].cpu().detach().numpy()
            except:
                embeddings = np.full((1, 768), np.nan, dtype=np.float32)
            return embeddings

        seen_captions = []
        text_embeddings = []
        for i, example in enumerate(dataset):
            text_embeddings.append(get_text_embeddings(example).squeeze())
            if i % 2500 == 0:
                print(f'{i}/{len(dataset)}', flush=True)
        ds_with_embeddings = dataset.add_column('text_embeddings', text_embeddings)
        print(f'Number of seen captions: {len(seen_captions)}')
        ds_with_embeddings.add_faiss_index(column='text_embeddings')

        for example in embeddings:
            scores, retrieved_examples = ds_with_embeddings.get_nearest_examples('text_embeddings',
                                                                                 example['caption_f'].numpy(), k=11)

            example['nearest_captions'] = retrieved_examples['caption'][1:]
            example['nearest_captions_f'] = [torch.tensor(e) for e in retrieved_examples['text_embeddings'][1:]]

        with open(os.path.join(args.output_dir, 'validation_features_dict_faiss.pth'), 'wb') as f:
            pickle.dump(embeddings, f)


def image_caption_benchmark_contrastive_faiss(args, device):
    if os.path.isfile(os.path.join(args.output_dir, 'validation_features_dict_faiss.pth')):
        with open(os.path.join(args.output_dir, 'validation_features_dict_faiss.pth'), 'rb') as f:
            embeddings = pickle.load(f)
    else:
        nearest_examples_dataset(args, device)
        with open(os.path.join(args.output_dir, 'validation_features_dict_faiss.pth'), 'rb') as f:
            embeddings = pickle.load(f)

    if args.sim_measure == 'ContraSim FAISS':
        encoder_path = os.path.join(args.output_dir, 'encoder_contrastive.pth')
        encoder = ContrastiveSim(768, args.out_dim, mid_layers=args.mid_layers).to(device)
    else:
        # Contrastive_dis
        encoder_path = os.path.join(args.output_dir, 'encoder_contrastive_dis.pth')
        encoder = ContrastiveSim_dis(768, args.out_dim, mid_layers=args.mid_layers).to(device)
    if os.path.isfile(encoder_path):
        checkpoint = torch.load(encoder_path, map_location=device)
        encoder.load_state_dict(checkpoint)
    else:
        with open(os.path.join(args.output_dir, f'train_features_torch.pth'), 'rb') as f:
            train_embeddings = pickle.load(f)
        train_embeddings = train_embeddings.to(device)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
        if args.sim_measure == 'ContraSim FAISS':
            criterion = SupConLoss(temperature=0.07).to(device)
        else:
            # Contrastive_dis
            criterion = SupConLossDis(temperature=0.07).to(device)

        for epoch in range(args.epochs):
            epoch_loss = 0
            num_batches = 0
            for batch in train_embeddings.split(args.train_batch_size):
                encoded_features = torch.empty(
                    (batch.shape[0], batch.shape[1], args.out_dim)).to(
                    device)  # [num_examples x num_views(2) x out_dim]
                for j, view in enumerate(batch.swapaxes(0, 1)):
                    encoded_features[:, j, :] = encoder.get_features(view)
                loss = criterion(encoded_features)
                epoch_loss += loss.item()
                num_batches += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'[{args.sim_measure}] Epoch: {epoch}, loss: {epoch_loss / num_batches:.3f}')

        torch.save(encoder.state_dict(), encoder_path)

    evaluate_faiss(args, embeddings, encoder)


def image_caption_benchmark_deep_faiss(args, device):
    if os.path.isfile(os.path.join(args.output_dir, 'validation_features_dict_faiss.pth')):
        with open(os.path.join(args.output_dir, 'validation_features_dict_faiss.pth'), 'rb') as f:
            embeddings = pickle.load(f)
    else:
        nearest_examples_dataset(args, device)
        with open(os.path.join(args.output_dir, 'validation_features_dict_faiss.pth'), 'rb') as f:
            embeddings = pickle.load(f)

    with open(os.path.join(args.output_dir, f'train_features_torch.pth'), 'rb') as f:
        train_embeddings = pickle.load(f)
    train_embeddings = train_embeddings.to(device)
    if args.sim_measure == 'DeepDot':
        encoder_path = os.path.join(args.output_dir, 'encoder_deep_dot.pth')
        encoder = deepDot(train_embeddings.shape[-1], args.out_dim, mid_layers=args.mid_layers).to(device)
    else:
        # DeepCKA
        encoder_path = os.path.join(args.output_dir, 'encoder_deep_cka.pth')
        encoder = deepCKA(train_embeddings.shape[-1], args.out_dim, mid_layers=args.mid_layers).to(device)
    encoder = get_encoder(args, train_embeddings, encoder, encoder_path, args.sim_measure)

    evaluate_faiss(args, embeddings, encoder)


def image_caption_benchmark_closed_form_faiss(args, device):
    with torch.no_grad():
        if os.path.isfile(os.path.join(args.output_dir, 'validation_features_dict_faiss.pth')):
            with open(os.path.join(args.output_dir, 'validation_features_dict_faiss.pth'), 'rb') as f:
                embeddings = pickle.load(f)
        else:
            nearest_examples_dataset(args, device)
            with open(os.path.join(args.output_dir, 'validation_features_dict_faiss.pth'), 'rb') as f:
                embeddings = pickle.load(f)

        evaluate_faiss(args, embeddings)


def evaluate_faiss(args, embeddings, encoder=None):
    with torch.no_grad():
        total, num_correct = 0, 0
        for batch_id, batch in enumerate(
                [embeddings[i:i + args.split_size] for i in range(0, len(embeddings), args.split_size)]):
            sim_list = []
            if args.sim_measure == 'CKA FAISS':
                true_image = np.stack([b['image_f'].numpy() for b in batch])
                true_sim = feature_space_linear_cka(true_image,
                                                    np.stack([b['caption_f'].numpy() for b in batch]))
                if np.isnan(true_sim):
                    print('Found nan')
                    exit(0)

            elif args.sim_measure == 'Dot FAISS':
                true_image = np.stack([b['image_f'].numpy() for b in batch])
                true_sim = torch.dot(torch.from_numpy(true_image).view(-1),
                                     torch.from_numpy(np.stack([b['caption_f'].numpy() for b in batch])).view(-1))
                if torch.isnan(true_sim):
                    print('Found nan')
                    exit(0)
                true_sim = true_sim.item()

            elif args.sim_measure == 'Norm FAISS':
                true_image = np.stack([b['image_f'].numpy() for b in batch])
                true_sim = (1 - (F.normalize(torch.from_numpy(true_image), dim=-1) -
                                 F.normalize(torch.from_numpy(np.stack([b['caption_f'].numpy() for b in batch])), dim=-1)).norm(dim=-1, p=2)).mean()
                if torch.isnan(true_sim):
                    print('Found nan')
                    exit(0)
                true_sim = true_sim.item()

            elif encoder is not None:
                true_image = torch.stack([b['image_f'] for b in batch]).to(device)
                true_sim = encoder(true_image, torch.stack([b['caption_f'] for b in batch]).to(device))
                if torch.isnan(true_sim):
                    print('Found nan')
                    exit(0)
                true_sim = true_sim.item()

            sim_list.append(true_sim)
            for i in range(10):
                if args.sim_measure == 'CKA FAISS':
                    current_embed = np.stack([b['nearest_captions_f'][i] for b in batch])
                    sim = feature_space_linear_cka(true_image,
                                                   current_embed)
                    if np.isnan(sim):
                        print('Found nan')
                        exit(0)

                elif args.sim_measure == 'Dot FAISS':
                    current_embed = np.stack([b['nearest_captions_f'][i] for b in batch])
                    sim = torch.dot(torch.from_numpy(true_image).view(-1), torch.from_numpy(current_embed).view(-1))
                    if torch.isnan(sim):
                        print('Found nan')
                        exit(0)
                    sim = sim.item()

                elif args.sim_measure == 'Norm FAISS':
                    current_embed = np.stack([b['nearest_captions_f'][i] for b in batch])
                    sim = (1 - (F.normalize(torch.from_numpy(true_image), dim=-1) -
                                     F.normalize(torch.from_numpy(current_embed), dim=-1)).norm(dim=-1, p=2)).mean()
                    if torch.isnan(sim):
                        print('Found nan')
                        exit(0)
                    sim = sim.item()

                elif encoder is not None:
                    current_embed = torch.stack([b['nearest_captions_f'][i] for b in batch]).to(device)
                    sim = encoder(true_image, current_embed)
                    if torch.isnan(sim):
                        print('Found nan')
                        exit(0)
                    sim = sim.item()


                sim_list.append(sim)
            sim_list = np.array(sim_list)
            if sim_list.argmax() == 0:
                num_correct += 1
            else:
                pass
            total += 1

        print(f'[{args.sim_measure}] accuracy = {num_correct / total * 100:.3f}%')


def image_caption_benchmark_closed_form(args, device):
    with torch.no_grad():
        if os.path.isfile(os.path.join(args.output_dir, 'validation_features_torch.pth')):
            with open(os.path.join(args.output_dir, f'validation_features_torch.pth'), 'rb') as f:
                embeddings = pickle.load(f)
        else:
            prepare_dataset(args.output_dir, device, args)
            with open(os.path.join(args.output_dir, f'validation_features_torch.pth'), 'rb') as f:
                embeddings = pickle.load(f)

        evaluate(args, embeddings)


def image_caption_benchmark_contrastive(args, device):
    if os.path.isfile(os.path.join(args.output_dir, 'validation_features_torch.pth')):
        with open(os.path.join(args.output_dir, 'validation_features_torch.pth'), 'rb') as f:
            embeddings = pickle.load(f)
    else:
        prepare_dataset(args.output_dir, device, args)
        with open(os.path.join(args.output_dir, 'validation_features_torch.pth'), 'rb') as f:
            embeddings = pickle.load(f)
    embeddings = embeddings.to(device)
    if args.sim_measure == 'ContraSim':
        encoder_path = os.path.join(args.output_dir, 'encoder_contrastive.pth')
        encoder = ContrastiveSim(embeddings.shape[-1], args.out_dim, mid_layers=args.mid_layers).to(device)
    else:
        encoder_path = os.path.join(args.output_dir, 'encoder_contrastive_dis.pth')
        encoder = ContrastiveSim_dis(embeddings.shape[-1], args.out_dim, mid_layers=args.mid_layers).to(device)
    if os.path.isfile(encoder_path):
        checkpoint = torch.load(encoder_path, map_location=device)
        encoder.load_state_dict(checkpoint)
    else:
        with open(os.path.join(args.output_dir, f'train_features_torch.pth'), 'rb') as f:
            train_embeddings = pickle.load(f)
        train_embeddings = train_embeddings.to(device)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
        if args.sim_measure == 'ContraSim':
            criterion = SupConLoss(temperature=0.07).to(device)
        else:
            criterion = SupConLossDis(temperature=0.07).to(device)
        for epoch in range(args.epochs):
            epoch_loss = 0
            num_batches = 0
            for batch in train_embeddings.split(args.train_batch_size):
                encoded_features = torch.empty(
                    (batch.shape[0], batch.shape[1], args.out_dim)).to(device)  # [num_examples x num_views(2) x out_dim]
                for j, view in enumerate(batch.swapaxes(0, 1)):
                    encoded_features[:, j, :] = encoder.get_features(view)
                loss = criterion(encoded_features)
                epoch_loss += loss.item()
                num_batches += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'[{args.sim_measure}] Epoch: {epoch}, loss: {epoch_loss / num_batches:.3f}')

        torch.save(encoder.state_dict(), encoder_path)

    evaluate(args, embeddings, encoder)


def get_encoder(args, train_embedding_vectors, encoder, encoder_name, sim_measure):
    if os.path.isfile(encoder_name):
        checkpoint = torch.load(encoder_name, map_location=device)
        encoder.load_state_dict(checkpoint)
    else:
        # Train it
        optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            epoch_loss = 0
            num_examples = 0
            for batch in train_embedding_vectors.split(args.train_batch_size):
                similarity = encoder(batch[:, 0, :], batch[:, 1, :])
                loss = -similarity
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_examples += 1
            print(f'[{sim_measure}] epoch: {epoch} loss: {epoch_loss / num_examples}')

        torch.save(encoder.state_dict(), encoder_name)

    return encoder


def evaluate(args, embeddings, encoder=None):
    with torch.no_grad():
        total, num_correct = 0, 0
        for batch in embeddings.split(args.split_size):
            sim_list = []
            if args.sim_measure == 'CKA':
                true_sim = feature_space_linear_cka(batch[:, 0, :].cpu().detach().numpy(),
                                                    batch[:, 1, :].cpu().detach().numpy())

            elif args.sim_measure == 'Dot product':
                true_sim = torch.dot(batch[:, 0, :].reshape(-1), batch[:, 1, :].reshape(-1))
                if torch.isnan(true_sim):
                    print('Found nan')
                    exit(0)
                true_sim = true_sim.item()

            elif args.sim_measure == 'Norm':
                true_sim = (1 - (F.normalize(batch[:, 0, :], dim=-1) -
                                 F.normalize(batch[:, 1, :], dim=-1)).norm(dim=-1, p=2)).mean()
                if torch.isnan(true_sim):
                    print('Found nan')
                    exit(0)
                true_sim = true_sim.item()

            elif encoder is not None:
                true_sim = encoder(batch[:, 0, :], batch[:, 1, :])
                if torch.isnan(true_sim):
                    print('Found nan')
                    exit(0)
                true_sim = true_sim.item()

            sim_list.append(true_sim)
            for i in range(10):
                temp_index = torch.randint(embeddings.shape[0], (batch.shape[0],))
                if args.sim_measure == 'CKA':
                    sim = feature_space_linear_cka(batch[:, 0, :].cpu().detach().numpy(),
                                                   embeddings[temp_index, 1, :].cpu().detach().numpy())
                    if np.isnan(sim):
                        print('Found nan')
                        exit(0)

                elif args.sim_measure == 'Dot product':
                    sim = torch.dot(batch[:, 0, :].reshape(-1), embeddings[temp_index, 1, :].reshape(-1))
                    if torch.isnan(sim):
                        print('Found nan')
                        exit(0)
                    sim = sim.item()

                elif args.sim_measure == 'Norm':
                    sim = (1 - (F.normalize(batch[:, 0, :], dim=-1) -
                                F.normalize(embeddings[temp_index, 1, :], dim=-1)).norm(dim=-1, p=2)).mean()
                    if torch.isnan(sim):
                        print('Found nan')
                        exit(0)
                    sim = sim.item()

                elif encoder is not None:
                    sim = encoder(batch[:, 0, :], embeddings[temp_index, 1, :])
                    if torch.isnan(sim):
                        print('Found nan')
                        exit(0)
                    sim = sim.item()

                sim_list.append(sim)

            sim_list = np.array(sim_list)
            if sim_list.argmax() == 0:
                num_correct += 1
            else:
                pass
            total += 1

        print(f'[{args.sim_measure}] accuracy = {num_correct / total * 100:.3f}%')


def image_caption_benchmark_deep(args, device):
    if os.path.isfile(os.path.join(args.output_dir, 'validation_features_torch.pth')):
        with open(os.path.join(args.output_dir, f'validation_features_torch.pth'), 'rb') as f:
            embeddings = pickle.load(f)
    else:
        prepare_dataset(args.output_dir, device, args)
        with open(os.path.join(args.output_dir, 'validation_features_torch.pth'), 'rb') as f:
            embeddings = pickle.load(f)
    embeddings = embeddings.to(device)
    with open(os.path.join(args.output_dir, f'train_features_torch.pth'), 'rb') as f:
        train_embeddings = pickle.load(f)
    train_embeddings = train_embeddings.to(device)
    encoder_path = os.path.join(args.output_dir, 'encoder_deep_cka.pth')
    if args.sim_measure == 'DeepCKA':
        encoder = deepCKA(embeddings.shape[-1], args.out_dim, mid_layers=args.mid_layers).to(device)
    else:
        encoder = deepDot(embeddings.shape[-1], args.out_dim, mid_layers=args.mid_layers).to(device)
    encoder = get_encoder(args, train_embeddings, encoder, encoder_path, args.sim_measure)

    evaluate(args, embeddings, encoder)


if __name__ == '__main__':
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device->{device}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--vision-model-list', type=list, default=['google/vit-base-patch16-224', 'facebook/convnext-tiny-224'])
    parser.add_argument('--text-model-list', type=list, default=['gpt2', 'bert-base-cased'])


    parser.add_argument('--num-of-test-embeddings', type=int, default=5000)
    parser.add_argument('--num-of-train-embeddings', type=int, default=10000)
    parser.add_argument('--vision-model', type=str)
    parser.add_argument('--text-model', type=str)
    parser.add_argument('--sim-measure', type=str)
    parser.add_argument('--output-dir', type=str, default='')
    parser.add_argument('--split-size', type=int, default=64)

    # Encoder configuration
    parser.add_argument('--out-dim', type=int, default=128)
    parser.add_argument('--train-batch-size', type=int, default=1024)
    parser.add_argument('--mid-layers', type=list, default=[512, 256])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)

    args = parser.parse_args()
    args.device = device
    print(args)
    signal.signal(signal.SIGALRM, timeout_handler)
    if not os.path.isdir('image_caption_benchmark'):
        os.mkdir('image_caption_benchmark')

    for vision_model in args.vision_model_list:
        args.vision_model = vision_model
        for text_model in args.text_model_list:
            args.text_model = text_model
            print('================================================================')
            print(f'Vision model: {vision_model}')
            print(f'Text model: {text_model}')
            print('================================================================')
            args.output_dir = os.path.join('image_caption_benchmark', f'{vision_model.split("/")[-1]}_{text_model.split("/")[-1]}')
            if not os.path.isdir(args.output_dir):
                os.mkdir(args.output_dir)
            args.sim_measure = 'CKA'
            image_caption_benchmark_closed_form(args, device)
            args.sim_measure = 'Dot product'
            image_caption_benchmark_closed_form(args, device)
            args.sim_measure = 'Norm'
            image_caption_benchmark_closed_form(args, device)
            args.sim_measure = 'DeepCKA'
            image_caption_benchmark_deep(args, device)
            args.sim_measure = 'DeepDot'
            image_caption_benchmark_deep(args, device)
            args.sim_measure = 'ContraSim'
            image_caption_benchmark_contrastive(args, device)
            args.sim_measure = 'ContraSim_dis'
            image_caption_benchmark_contrastive(args, device)


            args.sim_measure = 'Dot FAISS'
            image_caption_benchmark_closed_form_faiss(args, device)
            args.sim_measure = 'Norm FAISS'
            image_caption_benchmark_closed_form_faiss(args, device)
            args.sim_measure = 'CKA FAISS'
            image_caption_benchmark_closed_form_faiss(args, device)
            args.sim_measure = 'DeepDot FAISS'
            image_caption_benchmark_deep_faiss(args, device)
            args.sim_measure = 'DeepCKA FAISS'
            image_caption_benchmark_deep_faiss(args, device)
            args.sim_measure = 'ContraSim FAISS'
            image_caption_benchmark_contrastive_faiss(args, device)
            args.sim_measure = 'ContraSim_dis FAISS'
            image_caption_benchmark_contrastive_faiss(args, device)
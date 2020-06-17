#!/usr/bin/python3
# -*-coding:utf-8-*-
import os
import numpy as np
import json
import random
import cv2
import decord
import matplotlib.pyplot as plt
from config import Configuration


class BatchGenerator(object):
    def __init__(self, config):
        self._config = config
        self._samples = self.samples_statistic()
        self.visualize_samples()
        self._label_mapping = dict()
        self._train_samples, self._val_samples = self.split_train_val(train_val_ratio=5.0)

        self._train_batch_amount = len(self._train_samples) // self._config.batch_size
        self._val_batch_amount = len(self._val_samples) // self._config.batch_size
        print('Amount of training samples (may be augmented) = {}'.format(len(self._train_samples)))
        print('Amount of validation samples = {}'.format(len(self._val_samples)))
        print('Amount of training batches = {}'.format(self._train_batch_amount))
        print('Amount of validation batches = {}'.format(self._val_batch_amount))

        self._train_batch_index, self._val_batch_index = 0, 0

    def visualize_samples(self):
        labels_dist = dict()
        for s in self._samples:
            if s['gt'] in labels_dist.keys():
                labels_dist[s['gt']] += 1
            else:
                labels_dist[s['gt']] = 1

        by_value = sorted(labels_dist.items(), key=lambda x: -x[1])
        x, y = list(), list()
        for d in by_value:
            x.append(d[0])
            y.append(d[1])

        plt.figure(figsize=(12, 8))
        plt.bar(x, y, color='g')
        plt.xlabel('Label', fontdict={'size': 13})
        plt.xticks(x[0:-1:3], rotation=90)
        plt.ylabel('Samples Amount', fontdict={'size': 13})
        plt.title('Samples Distribution (# of classes = {})'.format(len(by_value)),
                  fontdict={
                      'size': 16
                  })
        plt.savefig('../samples_distribution.jpg', dpi=300, bbox_inches='tight')
        print('# of Labels = {}'.format(len(by_value)))

    def samples_statistic(self):
        samples = list()
        items = json.load(open(os.path.join(self._config.trainval_set_dir, 'lables.json'), 'r'))['lable']
        for item in items:
            video_name = list(item.keys())
            label = list(item.values())
            assert len(video_name) == 1
            assert len(label) == 1
            video_name = video_name[0]
            label = label[0]
            if not os.path.exists(os.path.join(self._config.trainval_set_dir, 'videos', video_name + '.mp4')):
                continue

            if {'video_name': video_name, 'gt': label} in samples:
                continue

            samples.append(
                {
                    'video_name': video_name,
                    'gt': label
                }
            )
        print('Total samples amount = {}'.format(len(samples)))
        return samples

    def split_train_val(self, train_val_ratio=5.0):
        labels = list(set([s['gt'] for s in self._samples]))
        for i, label in enumerate(labels):
            self._label_mapping[label] = i

        # dump to local disk for reference
        json.dump(self._label_mapping, open(self._config.mapping_path, 'w'), ensure_ascii=False, indent=True)

        random.shuffle(self._samples)
        batch_amount = round(len(self._samples) / self._config.batch_size)
        val_batch_amount = round(batch_amount * 1.0 / (1.0 + train_val_ratio))
        split_index = val_batch_amount * self._config.batch_size

        val_samples = self._samples[0:split_index]
        train_samples = self._samples[split_index:]

        train_batch_amount = int(np.ceil(len(train_samples) / self._config.batch_size))

        print('Training samples amount = {}, batch amount = {} (batch size = {})'.format(len(train_samples), train_batch_amount, self._config.batch_size))
        print('Validation samples amount = {}, batch amount = {} (batch size = {})'.format(len(val_samples), val_batch_amount, self._config.batch_size))

        aug_samples_amount = train_batch_amount * self._config.batch_size - len(train_samples)
        aug_part = list()
        for _ in range(aug_samples_amount):
            index = random.randint(0, len(train_samples)-1)
            aug_part.append(train_samples[index])

        train_samples.extend(aug_part)
        return train_samples, val_samples

    def analyze(self):
        for index, sample in enumerate(self._samples):
            video_path = os.path.join(self._config.trainval_set_dir, 'videos', sample['video_name'] + '.mp4')
            video = decord.VideoReader(video_path)
            print(index + 1, sample['video_name'], len(video), video[0].shape)

            frame = video[0]
            frame = frame.asnumpy()  # (height, width, channels)
            print(frame.shape)
            resized = cv2.resize(frame, (self._config.frame_width, self._config.frame_height))
            print(resized.shape)

    @staticmethod
    def sample_t_dimen(actual_frames, target_frames):
        index = np.linspace(0, actual_frames-1, target_frames)
        return index.astype(np.int)

    def next_train_batch(self):
        input_batch = np.zeros(shape=(self._config.batch_size,
                                      self._config.time_dimen,
                                      self._config.frame_height,
                                      self._config.frame_width,
                                      self._config.frame_channels))
        gt_batch = np.zeros(shape=(self._config.batch_size, self._config.ncls))

        for b_idx, sample in enumerate(self._train_samples[
                                       self._train_batch_index * self._config.batch_size:
                                       (1+self._train_batch_index) * self._config.batch_size]):
            video_path = os.path.join(self._config.trainval_set_dir, 'videos', sample['video_name'] + '.mp4')
            label = sample['gt']
            video = decord.VideoReader(video_path)
            sampled_frame_index_list = self.sample_t_dimen(len(video), target_frames=self._config.time_dimen)
            for t_idx, index in enumerate(sampled_frame_index_list):
                frame = video[index]
                frame = frame.asnumpy()     # (height, width, channels)
                resized_frame = cv2.resize(frame, (self._config.frame_width, self._config.frame_height))
                input_batch[b_idx][t_idx] = resized_frame

            gt_batch[b_idx][self._label_mapping[label]] = 1.0

        self._train_batch_index += 1
        return input_batch, gt_batch

    def next_val_batch(self):
        input_batch = np.zeros(
            shape=(
                self._config.batch_size,
                self._config.time_dimen,
                self._config.frame_height,
                self._config.frame_width,
                self._config.frame_channels))
        gt_batch = np.zeros(shape=(self._config.batch_size, self._config.ncls))

        for b_idx, sample in enumerate(self._val_samples[
                                       self._val_batch_index * self._config.batch_size:
                                       (1+self._val_batch_index) * self._config.batch_size]):
            video_path = os.path.join(self._config.trainval_set_dir, 'videos', sample['video_name'] + '.mp4')
            label = sample['gt']

            video = decord.VideoReader(video_path)
            sampled_frame_index_list = self.sample_t_dimen(len(video), target_frames=self._config.time_dimen)
            for t_idx, index in enumerate(sampled_frame_index_list):
                frame = video[index]
                frame = frame.asnumpy()     # (height, width, channels)
                resized_frame = cv2.resize(frame, (self._config.frame_width, self._config.frame_height))
                input_batch[b_idx][t_idx] = resized_frame

            gt_batch[b_idx][self._label_mapping[label]] = 1.0

        self._val_batch_index += 1
        return input_batch, gt_batch

    @property
    def train_batch_amount(self):
        return self._train_batch_amount

    @property
    def val_batch_amount(self):
        return self._val_batch_amount

    def reset_validation_batches(self):
        self._val_batch_index = 0

    def reset_training_batches(self):
        random.shuffle(self._train_samples)
        self._train_batch_index = 0


if __name__ == '__main__':
    batch_generator = BatchGenerator(
        config=Configuration()
    )

    for _ in range(batch_generator.train_batch_amount):
        train_batch, train_gt = batch_generator.next_train_batch()
        print(train_batch.shape, train_gt.shape)

    for _ in range(batch_generator.val_batch_amount):
        val_batch, val_gt = batch_generator.next_val_batch()
        print(val_batch.shape, val_gt.shape)

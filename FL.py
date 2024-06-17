import argparse
import os
import math
import numpy as np
import random
import time
import tomli
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from Training_time_attacks import attack_labelflipping, attack_backdoor
from Inference_time_attacks import attacks

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--configurations', help='experiment configurations')
    return parser.parse_args()

def load_configurations(config_path):
    with open(config_path, mode="rb") as fp:
        return tomli.load(fp)

def create_experiment_folder(exp_id, exp_description, work_space):
    exp_main_folder_path = work_space 
    training_time = datetime.now().strftime("%d-%m-%y_%I-%M-%S-%p") 
    exp_folder = os.path.join(exp_main_folder_path, f'Exp_{exp_id}_{training_time}_{exp_description}')
    os.mkdir(exp_folder)
    return exp_folder

def create_model(dataset_name, image_shape):
    model = Sequential()
    if dataset_name in ['mnist', 'fashion-mnist']:
        model.add(layers.Conv2D(32, (3, 3), padding="same", input_shape=image_shape))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(64, (3, 3), padding='same'))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512))
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10))
        model.add(layers.Activation("softmax"))
    elif dataset_name in ['cifar10', 'stl10']:
        model.add(layers.Conv2D(32, (3, 3), padding="same", input_shape=image_shape))
        model.add(layers.Activation("relu"))
        model.add(layers.Conv2D(32, (3, 3)))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (3, 3), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(512))
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10))
        model.add(layers.Activation("softmax"))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=3e-4,
        decay_steps=32,
        decay_rate=0.1
    )

    model.compile(
        optimizer='adam', 
        loss="categorical_crossentropy",
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

class Node:
    def __init__(self, id, name, node_type, node_level, saved_data_path, image_shape, n_classes):
        self.id = id
        self.name = name
        self.node_type = node_type
        self.node_level = node_level
        self.memory_path = f'level{self.node_level}/{self.name}/memory'
        self.data_path = f'{saved_data_path}/{self.name}/clean'
        self.global_model = ''
        self.local_model = ''
        self.local_model_num = 0
        self.training_epoch = 10
        self.training_batch_size = 32
        self.image_shape = image_shape
        self.n_classes = n_classes

    def set_training_parameters(self, training_epoch, training_batch_size):
        self.training_epoch = training_epoch
        self.training_batch_size = training_batch_size

    def print_info(self):
        print(f'Node ID: {self.id}, Name: {self.name}, Type: {self.node_type}, Level: {self.node_level}')
        print(f'Memory Path: {self.memory_path}, Global Model: {self.global_model}, Local Model: {self.local_model}')
        print(f'Training Epoch: {self.training_epoch}, Training Batch Size: {self.training_batch_size}')

    def get_clean_data(self):
        train_images = np.load(os.path.join(self.data_path, 'x_train.npy'))
        train_labels = np.load(os.path.join(self.data_path, 'y_train.npy'))
        return train_images, train_labels, len(train_images)

    def upload_model(self):
        return self.local_model

    def download_model(self, model):
        self.global_model = model

    def local_training(self, initial_training=True):
        model = create_model(self.image_shape, self.n_classes) if initial_training else tf.keras.models.load_model(self.global_model)
        train_images, train_labels, num_samples = self.get_clean_data()
        self.print_info()
        history = model.fit(x=train_images, y=train_labels, epochs=self.training_epoch, batch_size=self.training_batch_size)
        model_path = os.path.join(self.memory_path, 'model', 'local_model.h5')
        model.save(model_path)
        self.local_model = model_path
        return self.local_model, num_samples

class Client(Node):
    def __init__(self, id, name, node_type, node_level, saved_data_path, image_shape, n_classes):
        super().__init__(id, name, node_type, node_level, saved_data_path, image_shape, n_classes)
        self.DPA_enable = False
        self.MPA_enable = False
        self.train_images = []
        self.train_labels = []
        self.num_samples = 0

    def set_poisoning_parameters(self, DPA_enable, MPA_enable):
        self.DPA_enable = DPA_enable
        self.MPA_enable = MPA_enable

    def get_poisoned_data(self, train_images, train_labels, attack_name, percent_poison, target_labels, source_labels, continual_training_path):
        if attack_name == 'backdoor':
            _, poison_train_images, poison_train_labels = attack_backdoor.generate(train_images, train_labels, target_labels, source_labels, self.n_classes, percent_poison)
        elif attack_name == 'label_flipping':
            poison_train_images, poison_train_labels = attack_labelflipping.generate(train_images, train_labels)
        elif adversarial_training_enable:
            model = tf.keras.models.load_model(continual_training_path)
            poison_train_images, poison_train_labels = attacks.generate_poisoned_data(train_images, train_labels, model, attack_name)
            poison_train_images = np.concatenate((poison_train_images, train_images), axis=0)
            poison_train_labels = np.concatenate((poison_train_labels, train_labels), axis=0)
        return poison_train_images, poison_train_labels, len(poison_train_images)

    def local_training(self, initial_training=True, distillation_enable=False, global_round=1):
        model = create_model(self.image_shape, self.n_classes) if initial_training else tf.keras.models.load_model(self.global_model)
        self.print_info()
        if distillation_enable:
            teacher_model = tf.keras.models.load_model(teacher_model_path)
            preds = teacher_model.predict(x=self.train_images, batch_size=self.training_batch_size)
            self.train_labels = preds
        history = model.fit(x=self.train_images, y=self.train_labels, epochs=self.training_epoch, batch_size=self.training_batch_size)
        if self.MPA_enable:
            model = self.get_poisoned_model(model)
        model_path = os.path.join(self.memory_path, 'model', f'local_model_{global_round}.h5')
        model.save(model_path)
        self.local_model = model_path
        return self.local_model, self.num_samples

class Server(Node):
    def __init__(self, id, name, node_type, node_level, saved_data_path, image_shape, n_classes):
        super().__init__(id, name, node_type, node_level, saved_data_path, image_shape, n_classes)
        self.local_model_num = 1
        self.child_nodes = []
        self.aggregation_round = 1
        self.child_nodes_per_round = len(self.child_nodes)
        self.global_model_num = 1
        self.MPA_enable = False

    def set_child_nodes(self, child_nodes):
        self.child_nodes = child_nodes
        self.child_nodes_per_round = len(child_nodes)

    def aggregation(self, participants, global_round):
        w_locals = []
        model_path = self.global_model
        model_last = tf.keras.models.load_model(model_path)
        params_last = model_last.get_weights()
        array1 = np.array(params_last)
        sum_num_samples = 0
        for p in participants:
            p.download_model(self.global_model)
            if self.node_level == len(nodes) - 2:
                model_path, num_samples = p.local_training(initial_training=False, global_round=global_round)
            else:
                model_path, num_samples = p.start_aggregation(initial_training=False, continual_training=False, global_round=global_round)
            model_update = tf.keras.models.load_model(model_path)
            params = model_update.get_weights()
            array2 = np.array(params)
            array1 = array1 + (num_samples * array2)
            sum_num_samples += num_samples
            K.clear_session()
        array1 /= sum_num_samples
        model_update = tf.keras.models.load_model(model_path)
        model_update.set_weights(array1)
        if self.MPA_enable:
            model_update = self.get_poisoned_model(model_update)
        updated_model_path = os.path.join(self.memory_path, 'model', f'global_model_round_{global_round}.h5')
        model_update.save(updated_model_path)
        self.global_model = updated_model_path
        return self.global_model, sum_num_samples

    def start_aggregation(self, initial_training=True, continual_training=False, global_round=1):
        start_time = datetime.now()
        if initial_training:
            local_model, num_samples = self.local_training(initial_training)
            self.global_model = local_model
        if continual_training:
            local_model = continual_training_path
            self.global_model = local_model
        initial_training = False
        continual_training = False
        nonparticipant = self.child_nodes.copy()
        while global_round <= self.aggregation_round:
            participants = random.sample(nonparticipant, self.child_nodes_per_round)
            model, num_samples = self.aggregation(participants, global_round)
            global_round += 1
        model = tf.keras.models.load_model(self.global_model)
        model_path = os.path.join(self.memory_path, 'model', f'local_model_round_{self.local_model_num}.h5')
        model.save(model_path)
        self.local_model = model_path
        return self.local_model, num_samples

def main():
    prompt_args = parse_arguments()
    args = load_configurations(prompt_args.configurations)

    exp_id = args["experimnet_id"]
    saved_data_path = args["data_path"]
    work_space = args["store_folder"]
    dataset_name = args["dataset_name"]

    n_classes, image_shape = 10, (32, 32, 3) if dataset_name == 'cifar10' else 10, (28, 28, 1)
    num_clients = args["topology_description"]["number_of_clients"]
    topology_levels = args["topology_description"]["topology_levels"]
    topology = args["topology"]
    topology_name = args["topology_description"]["topology_name"]
    epochs = args["local_training"]["epochs"]
    batch_size = args["local_training"]["batch_size"]
    aggregation_round = args["federated_training"]["aggregation_round"]
    percent_of_participats = args["federated_training"]["percent_of_participats"]
    start_training = args["federated_training"]["start_training"]

    if start_training == 'start':
        initial_training = True
        continual_training = False
        start_from = 1
    else:
        initial_training = False
        continual_training = True
        continual_training_path = args["federated_training"]["resume_from_model"]
        start_from = args["federated_training"]["resume_from_round_number"]
        training_time = args["federated_training"]["resume_from_training_time"]

    defence_name = 'none'
    distillation_enable = args["distillation_training"]["distillation_enable"]
    if distillation_enable:
        teacher_model_path = args["distillation_training"]["teacher_model"]
        defence_name = 'distillation'

    adversarial_training_enable = args["adversarial_training"]["adversarial_training_enable"]
    if adversarial_training_enable:
        continual_training_path = args["federated_training"]["resume_from_model"]
        defence_name = 'adversarial_training'

    attack_enable = args["attack"]["attack_enable"]

    if attack_enable or adversarial_training_enable:
        DPA_list = ['backdoor', 'label_flipping']
        MPA_list = ['signflip']
        AT_list = ['fgm', 'Spatial_Transformation']
        attack_name = args["attack"]["attack_name"]
        adv_nodes = args["attack"]["malicious_nodes"]
        num_malicious_nodes = len(adv_nodes)
        if attack_name in DPA_list or attack_name in AT_list:
            DPA_enable = True
            MPA_enable = False
            percent_poison = args["attack"]["percent_of_poison_data"]
            source_labels = np.arange(n_classes)
            target_labels = (np.arange(n_classes) + 1) % n_classes
        elif attack_name in MPA_list:
            MPA_enable = True
            DPA_enable = False
    else:
        attack_name = 'none'
        num_malicious_nodes = 0
        adv_nodes = []

    exp_description = f'{topology_name}_{topology_levels}L_{dataset_name}_{num_clients}_attack_{attack_name}{num_malicious_nodes}_defence_{defence_name}'
    exp_folder = create_experiment_folder(exp_id, exp_description, work_space)
    os.chdir(exp_folder)
    
    print(f'Experiment Description: {exp_description}')
    print(f'Experiment Folder: {exp_folder}')

    nodes = [[] for _ in range(topology_levels)]

    def create_nodes(t, level, id):
        for parent, childs in t.items():
            if isinstance(childs, dict):
                nodes[level].append(Server(id=id, name=parent, node_type='server', node_level=level, saved_data_path=saved_data_path, image_shape=image_shape, n_classes=n_classes))
                id += 1
                create_nodes(childs, level + 1, id)
            else:
                nodes[level].append(Server(id=id, name=parent, node_type='server', node_level=level, saved_data_path=saved_data_path, image_shape=image_shape, n_classes=n_classes))
                id += 1
                for child in childs:
                    nodes[level + 1].append(Client(id=id, name=child, node_type='client', node_level=level + 1, saved_data_path=saved_data_path, image_shape=image_shape, n_classes=n_classes))
                    id += 1

    create_nodes(topology, 0, 0)

    for level in range(topology_levels):
        for node in nodes[level]:
            node.set_training_parameters(epochs, batch_size)
            if level != topology_levels - 1:
                node.aggregation_round = aggregation_round[level]
            if level == topology_levels - 2:
                for adv_node in adv_nodes:
                    if node.name == adv_node and MPA_enable:
                        node.MPA_enable = MPA_enable
            if level == topology_levels - 1:
                train_images, train_labels, num_samples = node.get_clean_data()
                for adv_node in adv_nodes:
                    if node.name == adv_node:
                        if DPA_enable:
                            node.DPA_enable = DPA_enable
                            train_images, train_labels, num_samples = node.get_poisoned_data(train_images, train_labels, attack_name, percent_poison, target_labels, source_labels, continual_training_path)
                        elif MPA_enable:
                            node.MPA_enable = MPA_enable
                node.train_images = train_images
                node.train_labels = train_labels
                node.num_samples = num_samples

    def set_child_nodes(t, level):
        for parent, childs in t.items():
            for node in nodes[level]:
                if node.name == parent:
                    parent_inx = nodes[level].index(node)
            for child in childs:
                for node in nodes[level + 1]:
                    if node.name == child:
                        child_inx = nodes[level + 1].index(node)
                nodes[level][parent_inx].set_child_nodes([nodes[level + 1][child_inx]])
            if isinstance(childs, dict):
                set_child_nodes(childs, level + 1)

    set_child_nodes(topology, 0)

    start_time = time.time()
    nodes[0][0].start_aggregation(initial_training=initial_training, continual_training=continual_training, global_round=1)
    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time}")

if __name__ == "__main__":
    main()

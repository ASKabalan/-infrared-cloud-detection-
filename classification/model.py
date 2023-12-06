# -*- coding: utf-8 -*-
# pylint: disable=W0223

from pathlib import Path
from typing import Dict

import jax
import optax
import orbax.checkpoint
from architecture import ResNet18, ResNet34
from flax.training import orbax_utils, train_state

print("devices used for jax and xla => " + f"{jax.default_backend()} / {jax.lib.xla_bridge.get_backend().platform}")

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------


@jax.jit
def update_model(state, batches_images, batches_labels, reg_l2=True):
    def loss_fn(params):
        logits, updates = state.apply_fn({"params": params, "batch_stats": state.batch_stats}, batches_images, mutable=["batch_stats"])
        loss_val = jax.numpy.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=batches_labels))

        if reg_l2:
            weight_penalty_params = jax.tree_util.tree_leaves(params)
            weight_decay = 0.0001
            weight_l2 = sum(jax.numpy.sum(x**2) for x in weight_penalty_params if x.ndim > 1)
            loss_val += weight_decay * 0.5 * weight_l2

        return loss_val, (logits, updates)

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = gradient_fn(state.params)
    new_state = state.apply_gradients(grads=grads, batch_stats=updates["batch_stats"])

    preds = jax.numpy.round(jax.nn.sigmoid(logits))
    accuracy = jax.numpy.mean(preds == batches_labels)

    return new_state, loss, accuracy


@jax.jit
def eval_function(state, batch_imgs_test, batch_labels):
    logits = state.apply_fn({"params": state.params, "batch_stats": state.batch_stats}, batch_imgs_test)
    loss = jax.numpy.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch_labels))
    preds = jax.numpy.round(jax.nn.sigmoid(logits))
    accuracy = jax.numpy.mean(preds == batch_labels)
    return loss, accuracy


@jax.jit
def pred_function(state, batch_imgs_test):
    logits = state.apply_fn({"params": state.params, "batch_stats": state.batch_stats}, batch_imgs_test)
    preds = jax.numpy.round(jax.nn.sigmoid(logits))
    return preds.astype(int)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------


class TrainState(train_state.TrainState):
    model_config: Dict
    batch_stats: Dict


RESNET_TYPES = {1: ResNet18, 2: ResNet34}
NB_CLASSES = 1
IMAGE_SHAPE = (512, 640)


def create_train_state(type_resnet, type_optimizer, nb_epochs, num_steps_per_epoch, momentum):
    model = RESNET_TYPES[type_resnet](momentum=momentum, output=None, n_classes=NB_CLASSES)
    variables = model.init(jax.random.PRNGKey(0), jax.numpy.ones([1, *IMAGE_SHAPE, 1]))
    schedule, optimizer = choice_of_optimiser(choice=type_optimizer, nb_epochs=nb_epochs, num_steps_per_epoch=num_steps_per_epoch)

    return (
        TrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            batch_stats=variables["batch_stats"],
            tx=optimizer,
            model_config={"TYPE_RESNET": type_resnet, "MOMENTUM": momentum},
        ),
        schedule,
    )


def save_model(state: TrainState, model_path: Path):
    orbax.checkpoint.PyTreeCheckpointer().save(model_path, {"model": state}, save_args=orbax_utils.save_args_from_target({"model": state}), force=True)


def load_model(model_path: Path):
    restored_data = orbax.checkpoint.PyTreeCheckpointer().restore(model_path)
    state = restored_data["model"]
    type_resnet = int(state["model_config"]["TYPE_RESNET"])
    momentum = int(state["model_config"]["MOMENTUM"])

    model = RESNET_TYPES[type_resnet](momentum=momentum, output=None, n_classes=NB_CLASSES)
    model.init(jax.random.PRNGKey(0), jax.numpy.ones([1, *IMAGE_SHAPE, 1]))
    _, optimizer = choice_of_optimiser(choice="piecewise", nb_epochs=0, num_steps_per_epoch=0)

    return TrainState.create(apply_fn=model.apply, params=state["params"], batch_stats=state["batch_stats"], tx=optimizer, model_config={"TYPE_RESNET": type_resnet})


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

INIT_VALUE = 1e-2


def choice_of_optimiser(choice: str, nb_epochs: int, num_steps_per_epoch: int):
    if choice == "piecewise":
        schedule = optax.piecewise_constant_schedule(
            init_value=INIT_VALUE,
            boundaries_and_scales={
                int(num_steps_per_epoch * nb_epochs * 0.3): 0.5,
                int(num_steps_per_epoch * nb_epochs * 0.6): 0.2,
                int(num_steps_per_epoch * nb_epochs * 0.85): 0.1,
            },
        )
    elif choice == "exponential":
        schedule = optax.exponential_decay(init_value=INIT_VALUE, transition_steps=50, decay_rate=0.8)

    return schedule, optax.sgd(learning_rate=schedule)

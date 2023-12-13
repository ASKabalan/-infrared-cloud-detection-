# -*- coding: utf-8 -*-
# pylint: disable=W0223

from pathlib import Path
from typing import Dict

import jax
import optax
import orbax.checkpoint
from architecture import ResNet18
from flax.training import orbax_utils, train_state

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------


@jax.jit
def update_model(state, batches_images, batches_labels, dropout_key):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        logits, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats}, batches_images, mutable=["batch_stats"], rngs={"dropout": dropout_train_key}
        )
        loss_val = jax.numpy.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=batches_labels))
        return loss_val, (logits, updates)

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = gradient_fn(state.params)
    new_state = state.apply_gradients(grads=grads, batch_stats=updates["batch_stats"])

    preds = jax.numpy.round(jax.nn.sigmoid(logits))
    accuracy = jax.numpy.mean(preds == batches_labels)

    return new_state, loss, accuracy


@jax.jit
def eval_function(state, batch_images, batch_labels):
    logits = state.apply_fn({"params": state.params, "batch_stats": state.batch_stats}, batch_images)
    loss = jax.numpy.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch_labels))
    preds = jax.numpy.round(jax.nn.sigmoid(logits))
    accuracy = jax.numpy.mean(preds == batch_labels)
    return loss, accuracy


@jax.jit
def pred_function(state, batch_images):
    logits = state.apply_fn({"params": state.params, "batch_stats": state.batch_stats}, batch_images)
    preds = jax.numpy.round(jax.nn.sigmoid(logits))
    return preds.astype(jax.numpy.int8)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------


class TrainState(train_state.TrainState):
    model_config: Dict
    batch_stats: Dict


NB_CLASSES = 1
IMAGE_SHAPE = (512, 640)


def create_train_state(type_optimizer, nb_epochs, nb_batch_train, momentum):
    root_key = jax.random.key(seed=0)
    params_key, dropout_key = jax.random.split(key=root_key, num=2)

    model = ResNet18(momentum=momentum, n_classes=NB_CLASSES)
    variables = model.init(params_key, jax.numpy.ones([1, *IMAGE_SHAPE, 1]))
    schedule, optimizer = choice_of_optimiser(choice=type_optimizer, nb_epochs=nb_epochs, nb_batch_train=nb_batch_train)

    return (
        TrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            batch_stats=variables["batch_stats"],
            tx=optimizer,
            model_config={"MOMENTUM": momentum},
        ),
        dropout_key,
        schedule,
    )


def save_model(state: TrainState, model_path: Path):
    orbax.checkpoint.PyTreeCheckpointer().save(
        model_path,
        {"model": state},
        save_args=orbax_utils.save_args_from_target({"model": state}),
        force=True,
    )


def load_model(model_path: Path):
    restored_data = orbax.checkpoint.PyTreeCheckpointer().restore(model_path)
    state = restored_data["model"]
    momentum = state["model_config"]["MOMENTUM"]
    model = ResNet18(momentum=momentum, n_classes=NB_CLASSES)
    model.init(jax.random.PRNGKey(0), jax.numpy.ones([1, *IMAGE_SHAPE, 1]))
    _, optimizer = choice_of_optimiser(choice="piecewise", nb_epochs=0, nb_batch_train=0)

    return TrainState.create(
        apply_fn=model.apply,
        params=state["params"],
        batch_stats=state["batch_stats"],
        tx=optimizer,
        model_config={"MOMENTUM": momentum},
    )


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

INIT_VALUE = 1e-2


def choice_of_optimiser(choice: str, nb_epochs: int, nb_batch_train: int):
    nb_steps = nb_batch_train * nb_epochs
    if choice == "piecewise":
        schedule = optax.piecewise_constant_schedule(
            init_value=INIT_VALUE,
            boundaries_and_scales={
                int(nb_steps * 1.0): 0.5,
                # int(nb_steps * 0.3): 0.5,
                # int(nb_steps * 0.6): 0.2,
                # int(nb_steps * 0.85): 0.1,
            },
        )
    elif choice == "exponential":
        schedule = optax.exponential_decay(init_value=INIT_VALUE, transition_steps=nb_steps, decay_rate=0.8)

    return schedule, optax.adam(learning_rate=schedule)

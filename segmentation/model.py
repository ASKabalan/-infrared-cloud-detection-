
import jax
from jax import jit, numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from flax.training import orbax_utils
import orbax.checkpoint
from typing import Any, Callable,Dict, Sequence, Tuple
import os
import shutil

class DoubleConvBlock(nn.Module):
    input_channels: int = 128
    kernel_size: Tuple[int, int] = (3 ,3)

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.input_channels, kernel_size=self.kernel_size , padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.input_channels, kernel_size=self.kernel_size , padding='SAME')(x)
        x = nn.relu(x)
        return x

class DownBlock(nn.Module):
    input_channels: int = 128

    @nn.compact
    def __call__(self, x):
        conv = DoubleConvBlock(self.input_channels)(x)
        x = nn.max_pool(conv, window_shape=(2,2), strides=(2,2), padding='SAME')
        return x , conv

class ExpansiveBlock(nn.Module):
    input_channels: int = 128

    @nn.compact
    def __call__(self, x , skip_connection):
        x = jax.image.resize(image=x, shape=(x.shape[0] , skip_connection.shape[1] , skip_connection.shape[2] , x.shape[3] ), method="nearest")
        x = jnp.concatenate([x ,skip_connection],axis=-1)
        x = DoubleConvBlock(self.input_channels)(x)
        return x

class CIRRUS_Net(nn.Module):
    input_channels: Sequence[int]
    bottle_neck_conv: int = 512
    output_channel: int=1
    @nn.compact
    def __call__(self, x):

        skip_outputs = []
        for conv_layer in self.input_channels:
            x , conv = DownBlock(conv_layer)(x)
            skip_outputs.append((conv_layer , conv))

        #x = nn.Conv(self.bottle_neck_conv, kernel_size=(3, 3))(x)
        #x = nn.relu(x)
        #x = jax.image.resize(x, (x.shape[0] * 2, x.shape[1] * 2, x.shape[2]) , method="nearest")

        for i , (unconv_layer , skip_output)  in enumerate(reversed(skip_outputs)):
            x = ExpansiveBlock(unconv_layer)(x,skip_output)

        x = nn.Conv(self.output_channel, kernel_size=(3, 3), padding='SAME')(x)
        return x

class TrainState(train_state.TrainState):
    model_config: Dict  # This is the configuration or any other information about the model


def create_train_state(rng, input_shape, input_channels = [64 , 128, 256 , 512], bottle_neck_conv = 1024, learning_rate = 1e-3, total_steps = 10):
    model = CIRRUS_Net(input_channels, bottle_neck_conv)
    params = model.init(rng, jnp.ones(input_shape, dtype=jnp.float32))['params']
    model_config = {"channels" : input_channels , "bottle_neck_conv" : bottle_neck_conv}
    # Initialize optimizer (Adam)
    cosine_decay_scheduler = optax.cosine_decay_schedule(learning_rate, decay_steps=total_steps, alpha=0.95)
    opt_adam = optax.adam(learning_rate=cosine_decay_scheduler)
    return TrainState.create(apply_fn=model.apply, params=params, tx=opt_adam,model_config=model_config)

@jit
def train_step(state, images, labels):
    print("apply_model Compiled once")
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    probs = jax.nn.sigmoid(logits)
    # Round to get binary predictions
    preds = jnp.round(probs)
    # Compute accuracy for binary classification
    accuracy = jnp.mean(preds == labels)
    new_state = state.apply_gradients(grads=grads)

    return new_state, loss, accuracy

@jit
def eval_step(state, images, labels):
    logits = state.apply_fn({'params': state.params}, images)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels))

    probs = jax.nn.sigmoid(logits)
    # Round to get binary predictions
    preds = jnp.round(probs)
    # Compute accuracy for binary classification
    accuracy = jnp.mean(preds == labels)

    return  loss, accuracy

@jit
def predict(params,apply_fn, images):
    logits = apply_fn({'params': params}, images)
    probs = jax.nn.sigmoid(logits)
    return probs

def save_model(state, model_path,checkpoint):
    """
    Save the Flax model using orbax.

    Parameters:
    - state: Flax train_state to be saved.
    - model_path: Path where the model will be saved.
    """
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target({'model': state})
    target_dir = f"{model_path}/model_{checkpoint}"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    orbax_checkpointer.save(f"{model_path}/model_{checkpoint}", {'model': state}, save_args=save_args)

def load_model(model_path,ModelClass):
    """
    Load the Flax model using orbax.

    Parameters:
    - model_path: Path from where the model will be loaded.

    Returns:
    - state: Loaded Flax train_state.
    """
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored_data = orbax_checkpointer.restore(model_path)
    state = restored_data['model']

    model_config = state['model_config']
    model_instance = ModelClass(**model_config)

    return state , model_instance.apply
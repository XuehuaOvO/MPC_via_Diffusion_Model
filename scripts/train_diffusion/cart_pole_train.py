import os
import torch
import wandb
import git

from experiment_launcher import single_experiment_yaml, run_experiment
from mpd import trainer
from mpd.models import UNET_DIM_MULTS, ConditionedTemporalUnet
from mpd.trainer import get_dataset, get_model, get_loss, get_summary
from mpd.trainer.trainer import get_num_epochs
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device

# from classifier_free_guidance_pytorch import TextConditioner
# from classifier_free_guidance_pytorch import classifier_free_guidance_class_decorator

@single_experiment_yaml
def experiment(
    ########################################################################
    # Dataset
    dataset_subdir: str = 'CartPole-NMPC',
    include_velocity: bool = False,

    ########################################################################
    # Diffusion Model
    diffusion_model_class: str = 'GaussianDiffusionModel',
    variance_schedule: str = 'exponential',  # cosine
    n_diffusion_steps: int = 25,
    predict_epsilon: bool = True,

    # Unet
    unet_input_dim: int = 32,
    unet_dim_mults_option: int = 1,

    ########################################################################
    # Loss
    loss_class: str = 'GaussianDiffusionCartPoleLoss',

    # Training parameters
    batch_size: int = 32,
    lr: float = 1e-4,
    num_train_steps: int = 5000, # 50000

    use_ema: bool = True,
    use_amp: bool = False,

    # model saving address
    model_saving_address = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/data_trained_models',

    # Summary parameters
    steps_til_summary: int = 2000,
    summary_class: str = 'SummaryTrajectoryGeneration',

    steps_til_ckpt: int = 5000,

    ########################################################################
    device: str = 'cuda',

    debug: bool = True,

    ########################################################################
    # MANDATORY
    seed: int = 0,
    results_dir: str = 'logs',

    ########################################################################
    # WandB
    wandb_mode: str = 'disabled',  # "online", "offline" or "disabled"
    wandb_entity: str = 'scoreplan',
    wandb_project: str = 'test_train',
    **kwargs
):
    fix_random_seed(seed)

    device = get_torch_device(device=device)
    print(f'device --{device}')
    tensor_args = {'device': device, 'dtype': torch.float32}

    # Dataset
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='InputsDataset',
        include_velocity=include_velocity,
        dataset_subdir=dataset_subdir,
        batch_size=batch_size,
        results_dir=results_dir,
        save_indices=True,
        tensor_args=tensor_args
    )

    print(f'train_subset -- {len(train_subset.indices)}')
    dataset = train_subset.dataset
    print(f'dataset -- {dataset}')

    # # texts
    # repo = git.Repo('.', search_parent_directories=True)
    # dataset_base_dir = os.path.join(repo.working_dir, 'data_trajectories')
    # base_dir = os.path.join(dataset_base_dir, dataset_subdir)
    # condition_texts = torch.load(os.path.join(base_dir, 'x-collecting_list_6400.pt'),map_location=tensor_args['device'])


    # Model
    diffusion_configs = dict(
        variance_schedule=variance_schedule,
        n_diffusion_steps=n_diffusion_steps,
        predict_epsilon=predict_epsilon,
    )

    unet_configs = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
        unet_input_dim=unet_input_dim,
        dim_mults=UNET_DIM_MULTS[unet_dim_mults_option],
    )

    model = get_model(
        model_class=diffusion_model_class,
        model=ConditionedTemporalUnet(**unet_configs), # TemporalUnet(**unet_configs)
        tensor_args=tensor_args,
        **diffusion_configs,
        **unet_configs
        # text_condition_type = 'film', 
        # text_condition_model_types = ('t5'),
        # text_condition_hidden_dims = (32, 64, 128),
        # text_condition_cond_drop_prob = 0.25,
    )

    # Loss
    loss_fn = val_loss_fn = get_loss(
        loss_class=loss_class
    )

    # text conditioner
    # text_conditioner = TextConditioner(
    # model_types = 't5',    
    # hidden_dims = (32,64,128),
    # hiddens_channel_first = False,
    # cond_drop_prob = 0.25  # conditional dropout 20% of the time, must be greater than 0. to unlock classifier free guidance
    # ).cuda()

    # Summary
    # summary_fn = get_summary(
    #     summary_class=summary_class,
    # )

    # Train
    trainer.train(
        model=model,
        train_dataloader=train_dataloader,
        train_subset=train_subset,
        val_dataloader=val_dataloader,
        val_subset=train_subset,
        epochs=get_num_epochs(num_train_steps, batch_size, len(dataset)),
        model_dir=results_dir,
        # summary_fn=summary_fn,
        lr=lr,
        loss_fn=loss_fn,
        val_loss_fn=val_loss_fn,
        steps_til_summary=steps_til_summary,
        steps_til_checkpoint=steps_til_ckpt,
        clip_grad=True,
        use_ema=use_ema,
        use_amp=use_amp,
        debug=debug,
        model_saving_address = model_saving_address,
        #text_conditioner = text_conditioner,
        tensor_args=tensor_args
    )

    # print(f'DEBUG MODE: {debug}')

    # print(f'kwargs: {kwargs}')

    # filename = os.path.join(results_dir, 'log_' + str(seed) + '.txt')
    # out_str = f'Running experiment with seed {seed}, env {env}, ' \
    #           f'env_param {env_param}, a {a}, ' \
    #           f'boolean_param {boolean_param}, some_default_param {some_default_param}'
    # print(out_str)
    # with open(filename, 'w') as file:
    #     file.write('Some logs in a log file.\n')
    #     file.write(out_str)

    # wandb.log({'seed': seed}, step=1)


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)

import argparse
import torch
import numpy as np
import json
import logging
import warnings
import os
from os.path import join, split
from util import utils
from datasets.CameraPoseDataset import CameraPoseDataset
from models.pose_losses import CameraPoseLoss
from models.transposenet.EMSTransPoseNet import EMSTransPoseNet
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_name",
                            help="name of model to create")
    arg_parser.add_argument("--mode", help="train or test")
    arg_parser.add_argument("--backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("--dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("--scene", help="scene name")
    arg_parser.add_argument("--labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("--config_file", help="path to configuration file", default="7scenes-config.json")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("--experiment", default="exp", help="experiment name")
    arg_parser.add_argument('--gpus', nargs="+", type=int, default=[0], help='device numbers of gpus to use')

    args = arg_parser.parse_args()

    # Set log
    utils.init_logger(args)

    # Record execution details
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    logging.info("GPU: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    logging.info("Start {} with {}".format(args.model_name, args.mode))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.model_name]
    general_params = config['general']
    config = {**model_params, **general_params}
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Create the model
    model = EMSTransPoseNet(config, args.backbone_path).to(device)

    if args.mode == 'train':
        writer = SummaryWriter(utils.create_output_dir(join('out', args.experiment, args.mode, args.scene)))
        
        # Load the checkpoint if needed
        if args.checkpoint_path:
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
            logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

        # Freeze parts of the model if indicated
        freeze = config.get("freeze")
        freeze_exclude_phrase = config.get("freeze_exclude_phrase")
        if isinstance(freeze_exclude_phrase, str):
            freeze_exclude_phrase = [freeze_exclude_phrase]
        if freeze:
            for name, parameter in model.named_parameters():
                freeze_param = True
                for phrase in freeze_exclude_phrase:
                    if phrase in name:
                        freeze_param = False
                        break
                if freeze_param:
                        parameter.requires_grad_(False)

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)
        nll_loss = torch.nn.NLLLoss()
        lambda_aux = config.get('lambda_aux')

        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        # Set the dataset and data loader
        no_augment = config.get("no_augment")
        if no_augment:
            transform = utils.test_transforms.get('baseline')
        else:
            transform = utils.train_transforms.get('baseline')

        equalize_scenes = config.get("equalize_scenes")
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform, equalize_scenes)
        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        checkpoint_dir = join('out', args.experiment, 'ckpts')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")
        n_batch = len(dataloader)

        # Set to train mode
        model.train()

        # Train
        for epoch in range(n_epochs):
            # Resetting temporal loss used for logging
            total_loss_x = 0.0
            total_loss_q = 0.0
            total_loss = 0.0
            n_samples = 0
            n_total_samples = n_batch * epoch

            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                gt_scene = minibatch.get('scene').to(device)
                batch_size = gt_pose.shape[0]
                n_samples += batch_size

                if freeze: # For TransPoseNet
                    model.eval()
                    with torch.no_grad():
                        transformers_res = model.forward_transformers(minibatch)
                    model.train()

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass to estimate the pose
                if freeze:
                    res = model.forward_heads(transformers_res)
                else:
                    res = model(minibatch)

                est_pose = res.get('pose')
                est_scene_log_distr = res.get('scene_log_distr')
                res_mean_t, res_mean_rot = res.get('res_mean_t'), res.get('res_mean_rot')
                
                # Pose Loss + Scene Loss
                loss_x, loss_q, loss = pose_loss(est_pose, gt_pose)
                scene_loss = nll_loss(est_scene_log_distr, gt_scene)
                # Query-Key Alignment Loss
                aux_loss_x = res_mean_t.mean()
                aux_loss_q = res_mean_rot.mean()
                loss = loss + scene_loss + lambda_aux * (aux_loss_x + aux_loss_q)

                # Collect for recoding and plotting
                total_loss_x += loss_x.item()
                total_loss_q += loss_q.item()
                total_loss += loss.item()

                # Back prop
                loss.backward()
                optimizer.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    logging.info("Epoch [{}/{}] Batch [{}/{}] Loss: {:.3f}".format(
                        epoch+1, n_epochs, batch_idx+1, n_batch, (total_loss/n_samples)))
            
            # Record loss and performance on train set
            writer.add_scalar('Train/Loss', total_loss, epoch)
            writer.add_scalar('Train/LossX', total_loss_x, epoch)
            writer.add_scalar('Train/LossQ', total_loss_q, epoch)

            # Save
            if (epoch % n_freq_checkpoint) == 0:
                torch.save(model.state_dict(), join(checkpoint_dir, f'{epoch+1}.pth'))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')

    else: # Test
        # Set the dataset and data loader
        labels_file = split(args.labels_file)[-1]
        transform = utils.test_transforms.get('baseline')
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform)
        n_img = len(dataset)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Load the checkpoint
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

        # Set to eval mode
        model.eval()

        pred = torch.zeros((n_img, 7))
        gt = torch.Tensor(dataloader.dataset.poses)
        
        with torch.no_grad():
            pbar = tqdm(dataloader, leave=False, desc="Test")
            for i, minibatch in enumerate(pbar, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                minibatch['scene'] = None # avoid using ground-truth scene during prediction
                output = model(minibatch)
                pred[i] = output.get('pose').detach().cpu()

        x_err, q_err = utils.pose_err(pred, gt)

        # Record overall statistics
        logging.info("Performance of {} on {}".format(args.checkpoint_path, labels_file))
        logging.info("Median pose error: {:.2f}m, {:.2f}deg\n".format(x_err, q_err))
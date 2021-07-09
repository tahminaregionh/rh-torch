import torch
import torchio as tio
from torch.utils.data import DataLoader
from torchio.data import GridSampler, GridAggregator
from rhtorch.config_utils import UserConfig
from rhtorch.utilities.modules import recursive_find_python_class
import numpy as np
from pathlib import Path
import argparse
import nibabel as nib
import sys
from tqdm import tqdm
import os


def infer_data_from_model(model, subject, ps=None, po=None, bs=1, GPU=True):
    """Infer a full volume given a trained model for 1 patient

    Args:
        model (torch.nn.Module): trained pytorch model
        subject (torchio.Subject): Subject instance from TorchIO library
        ps (list, optional): Patch size (from config). Defaults to None.
        po (int or list, optional): Patch overlap. Defaults to None.
        bs (int, optional): batch_size (from_config). Defaults to 1.

    Returns:
        [np.ndarray]: Full volume inferred from model
    """
    grid_sampler = GridSampler(subject, ps, po)
    patch_loader = DataLoader(grid_sampler, batch_size=bs)
    aggregator = GridAggregator(grid_sampler)
    with torch.no_grad():
        for patches_batch in patch_loader:
            patch_x, _ = model.prepare_batch(patches_batch)
            if GPU:
                patch_x = patch_x.to('cuda')
            locations = patches_batch[tio.LOCATION]
            patch_y = model(patch_x)
            aggregator.add_batch(patch_y, locations)
    return aggregator.get_output_tensor()


def save_nifty(data, ref, filename_out):
    func = nib.load(ref).affine
    data = data.squeeze().cpu().detach().numpy()
    ni_img = nib.Nifti1Image(data, func)
    nib.save(ni_img, filename_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Infer new data from input model.')
    parser.add_argument("-i", "--input",
                        help="Model directory. Should contain config.yaml file and model checkpoints. Will use current working directory if nothing passed",
                        type=str, default=os.getcwd())
    parser.add_argument("-c", "--config",
                        help="Config file of saved model",
                        type=str, default='config.yaml')
    parser.add_argument("-t", "--test",
                        help="Only use a subset of the valid dataset.",
                        action="store_true", default=False)
    parser.add_argument("--CPU",
                        help="Only use CPU",
                        action="store_true", default=False)

    args = parser.parse_args()
    model_dir = Path(args.input)
    test = args.test

    # load configs in inference mode
    user_configs = UserConfig(model_dir, arguments=args, mode='infer')
    configs = user_configs.hparams

    project_dir = Path(configs['project_dir'])
    project_id = project_dir.name
    data_dir = Path(configs['data_folder'])
    target_filename = configs['target_files']['name'][0]
    model_dir = Path(configs['model_dir'])
    model_name = configs['model_name']
    infer_dir = model_dir.joinpath('inferences')
    infer_dir.mkdir(parents=True, exist_ok=True)
    data_shape_in = configs['data_shape_in']
    patch_size = configs['patch_size']
    patch_overlap = int(np.min(patch_size) / 2)

    # load the test data
    sys.path.insert(1, str(project_dir))
    import data_generator
    data_gen = getattr(data_generator, configs['data_generator'])
    data_module = data_gen(configs)
    data_module.prepare_data()
    data_module.setup(stage='test')

    # load the model
    module_name = recursive_find_python_class(configs['module'])
    model = module_name(configs, data_shape_in)
    # Load the final (best) model
    if 'best_model' in configs:
        ckpt_path = Path(configs['best_model'])
        epoch_suffix = ''
    # Not done training. Load the most recent (best) ckpt
    else:
        ckpt_path = project_dir.joinpath('trained_models',
                                         model_name,
                                         'checkpoints',
                                         'Checkpoint_min_val_loss-v2.ckpt')
        epoch_suffix = None
    ckpt = torch.load(ckpt_path)
    if epoch_suffix is None:
        epoch_suffix = '_e={}'.format(ckpt['epoch'])
    model.load_state_dict(ckpt['state_dict'])
    if not args.CPU:
        model.cuda()
    model.eval()

    for patient in tqdm(data_module.test_set):
        patient_id = patient.id
        out_subdir = infer_dir.joinpath(patient_id)
        out_subdir.mkdir(parents=True, exist_ok=True)
        output_file = out_subdir.joinpath(
            f'Inferred_{model_name}{epoch_suffix}.nii.gz')

        # check if recontruction already done
        if not output_file.exists():
            # full volume inference - returns np.ndarray
            full_volume = infer_data_from_model(
                model, patient, patch_size, patch_overlap, GPU=not args.CPU)

            """ Below shows several ways to save the output """
            ref_nifty_file = data_dir.joinpath(
                patient_id).joinpath(target_filename)

            # save as nifty with nibabel - can add saving as .npy as well
            save_nifty(full_volume, ref_nifty_file, output_file)

            # save nifty with torchio
            tio.ScalarImage(tensor=full_volume,
                            affine=patient.input0.affine).save(output_file)

            # When any transformation is performed as part of
            # setup(stage='test'), we need to invert these. Do this by adding
            # the inferred image to the patient tio.Subject and invert.
            # Not all transformations can be inverted. We will resample to the
            # reference to make sure e.g. the voxel spacing are correct.
            temp = tio.ScalarImage(tensor=torch.rand(1, 1, 1, 1),
                                   affine=patient.input0.affine)
            patient.add_image(temp, 'predicted')
            patient.predicted.set_data(full_volume)
            patient.add_image(tio.ScalarImage(ref_nifty_file), 'reference')
            resample = tio.Resample('reference')
            patient_native_space = resample(patient.apply_inverse_transform())
            # Apply de-normalization using inverse of preproccess transform on
            # the target image. "inv_<proprocess_step[0]>" must be defined
            # in data_module.get_normalization_transform
            patient_native_space = data_module.de_normalize(
                patient_native_space,
                configs['target_files']['preprocess_step'][0])
            patient_native_space.predicted.save(output_file)

        else:
            print(f'Data already reconstructed with model {model_name}')

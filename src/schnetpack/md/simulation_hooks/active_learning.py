import json
import os
from sgdml.utils.io import dataset_md5
from sgdml import __version__
from sgdml.cli import create, train, validate, select, test
from sgdml.utils import io, ui
from turtle import position
import multiprocessing as mp

import h5py
import numpy as np
import torch

from schnetpack.md.simulation_hooks import SimulationHook
import schnetpack as spk
from schnetpack.md.parsers.md_setup import SetupCalculator


__all__ = ["Active_Learning_sGDML"]

def _create_task_from_model(model, dataset, sig):
        idxs_train = dataset['idxs_train']
        R_train = dataset['R'][idxs_train, :, :]
        F_train = dataset['F'][idxs_train, :, :]

        use_E = 'e_err' in model
        use_E_cstr = 'alphas_E' in model
        use_sym = model['perms'].shape[0] > 1

        task = {
            'type': 't',
            'code_version': __version__,
            'dataset_name': model['dataset_name'],
            'dataset_theory': model['dataset_theory'],
            'z': model['z'],
            'R_train': R_train,
            'F_train': F_train,
            'idxs_train': idxs_train,
            'md5_train': dataset['md5'],
            'idxs_valid': model['idxs_valid'],
            'md5_valid': dataset['md5'],
            'n_test': model['n_test'],
            'md5_test':dataset['md5'],
            'sig': sig,
            'lam': model['lam'],
            'use_E': model['use_E'],
            'use_E_cstr': use_E_cstr,
            'use_sym': use_sym,
            'perms': model['perms'],
            'use_cprsn': model['use_cprsn'],
            'solver_name': model['solver_name'],
            'solver_tol': model['solver_tol'], # check me
            'n_inducing_pts_init': model['n_inducing_pts_init'],
            'interact_cut_off': None, # check me
        }

        if use_E:
            task['E_train'] = dataset['E'][idxs_train]

        if 'lattice' in model:
            task['lattice'] = model['lattice']

        if 'r_unit' in model and 'e_unit' in model:
            task['r_unit'] = model['r_unit']
            task['e_unit'] = model['e_unit']

        if 'alphas_F' in model:
            task['alphas0_F'] = model['alphas_F']

        if 'alphas_E' in model:
            task['alphas0_E'] = model['alphas_E']

        if 'solver_iters' in model:
            task['solver_iters'] = model['solver_iters']

        if 'inducing_pts_idxs' in model:
            task['inducing_pts_idxs'] = model['inducing_pts_idxs']

        return task



class Active_Learning_sGDML(SimulationHook):
    def __init__(self,  true_calculator, error_check_func, system, path_to_new_dataset: str, path_to_sgdml_model: str, path_to_sgdml_traning_dataset: str, max_number_traj_in_mem = 5000) -> None:
        super(Active_Learning_sGDML, self).__init__()
        self.path_to_new_dataset = path_to_new_dataset
        self.calculator = true_calculator
        self.error_check_func = error_check_func
        self.path_to_sgdml_model = path_to_sgdml_model
        self.path_to_sgdml_traning_dataset = path_to_sgdml_traning_dataset
        self.current_number_of_bad_points = 0
        self.number_of_files = 0
        self.max_number_traj_in_mem = max_number_traj_in_mem
        self.z = system.atom_types.to('cpu').detach().numpy().flatten()
        self.n_atoms = len(self.z)
        self.bad_point_R_array = np.zeros((max_number_traj_in_mem, self.n_atoms, 3))
        self.bad_point_F_array = np.zeros((max_number_traj_in_mem, self.n_atoms, 3))
        self.bad_point_E_array = np.zeros((max_number_traj_in_mem, 1))

    def on_step_middle(self, simulator):
        self.on_simulation_start(simulator)

    def on_simulation_start(self, simulator):
        system = simulator.system
        inputs = self.calculator._generate_input(system)
        results = self.calculator.model(inputs)

        if self.error_check_func(system.forces/self.calculator.force_conversion, results['forces']):
            if  self.current_number_of_bad_points < self.max_number_traj_in_mem:
                print ('add point ...')
                self._add_point_to_bad_point_arrays(simulator, results)
                if simulator.step == simulator.n_steps - 1:
                    self._save_current_bad_point_arrays()
            else:
                self._save_current_bad_point_arrays()
        else:
            print ('no need to add point')


    def _save_current_bad_point_arrays(self):
        print ('save ...')
        train_dataset = dict(np.load(self.path_to_sgdml_traning_dataset,allow_pickle=True))
        new_dataset_path = self.path_to_new_dataset.split('.npz')[0] +'_'+str(self.number_of_files)+'_bad_points.npz'
        # if os.path.exists(new_dataset_path):
        #     new_dataset_path = self.path_to_sgdml_traning_dataset.split('.npz')[0]+'_'+str(self.number_of_files)+'_bad_points.npz'
        train_dataset['R'] = self.bad_point_R_array
        train_dataset['F'] = self.bad_point_F_array
        train_dataset['E'] = self.bad_point_E_array
        train_dataset['md5'] = dataset_md5(train_dataset)
        np.savez_compressed(new_dataset_path, **train_dataset)
        self.current_number_of_bad_points = 0
        self.bad_point_R_array *= 0
        self.bad_point_F_array *= 0
        self.bad_point_E_array *= 0
        self.number_of_files +=1


    def _add_point_to_bad_point_arrays(self, simulator, true_results):
        self.bad_point_R_array[self.current_number_of_bad_points] = (simulator.system.positions[0] * self.calculator.position_conversion).to('cpu').detach().numpy()
        self.bad_point_F_array[self.current_number_of_bad_points] = (true_results['forces']*self.calculator.force_conversion).to('cpu').detach().numpy()
        self.bad_point_E_array[self.current_number_of_bad_points] = (true_results['energy']).to('cpu').detach().numpy()
        self.current_number_of_bad_points += 1
        # pass

    def _add_point_to_traning_dataset(self, simulator, true_results):
        model = dict(np.load(self.path_to_sgdml_model,allow_pickle=True))
        train_dataset = dict(np.load(self.path_to_sgdml_traning_dataset,allow_pickle=True))

        idxs_train_old = model['idxs_train']
        R_train_old = train_dataset['R']
        E_train_old = train_dataset['E']
        F_train_old = train_dataset['F']

        R_new_point = (simulator.system.positions[0] * self.calculator.position_conversion).to('cpu').detach().numpy()
        E_new_point = (true_results['energy']).to('cpu').detach().numpy()
        F_new_point = (true_results['forces']*self.calculator.force_conversion).to('cpu').detach().numpy()

        idxs_train = np.concatenate([idxs_train_old, np.array([len(R_train_old)])])
        R_train = np.concatenate([R_train_old,R_new_point])
        E_train = np.concatenate([E_train_old,E_new_point])
        F_train = np.concatenate([F_train_old,F_new_point])

        train_dataset['R'] = R_train
        train_dataset['E'] = E_train
        train_dataset['F'] = F_train
        train_dataset['idxs_train'] = idxs_train
        # print (model.keys())
        # base_vars['md5'] = dataset_md5(base_vars)
        # print (base_vars['md5'])
        new_dataset_path = self.path_to_sgdml_traning_dataset.split('.n')[0]+'_'+str(len(idxs_train))+'.npz'
        np.savez_compressed(new_dataset_path, **train_dataset)
        return new_dataset_path

    def train_new_sgdml_model(self, path_to_dataset,simulator):
        overwrite = False
        use_torch = False
        max_processes = mp.cpu_count()
        model = np.load(self.path_to_sgdml_model, allow_pickle=True)
        dataset = np.load(path_to_dataset, allow_pickle=True)

        cwd = os.getcwd()
        task_dir = os.path.join(cwd, 'saves')
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
        sigmas = list(range(model['sig'] - 50,model['sig'] + 50, 10))
        for sig in sigmas:
            task = _create_task_from_model(model, dataset, sig)
            if sig < 100:
                sig_name = '0'+str(sig)
            else:
                sig_name = str(sig)
            task_path = os.path.join(task_dir,'task_'+sig_name+'.npz')
            np.savez_compressed(task_path, **task)


        task_dir_arg = io.is_dir_with_file_type(task_dir, 'task', or_file=True)
        print ('task_dir_arg',task_dir_arg)
        model_dir_or_file_path = train(
                                        task_dir_arg,
                                        valid_dataset = (path_to_dataset,dataset),
                                        overwrite = overwrite,
                                        max_processes = max_processes,
                                        use_torch=use_torch)
        # print ('model_dir_or_file_path',model_dir_or_file_path)
        model_dir_arg = io.is_dir_with_file_type(
            model_dir_or_file_path, 'model', or_file=True
        )

        ui.print_step_title('STEP 3', 'Hyper-parameter selection')
        model_file_name = select(
            model_dir_arg, overwrite, max_processes, None#, **kwargs
        )

        ui.print_step_title('STEP 4', 'Testing')
        model_dir_arg = io.is_dir_with_file_type(model_file_name, 'model', or_file=True)
        test(
            model_dir_arg,
            test_dataset=(path_to_dataset,dataset),
            n_test = task['n_test'],
            overwrite=False,
            max_processes=max_processes,
            use_torch=use_torch,
            #**kwargs
        )

        print(
            '\n'
            + ui.color_str('  DONE  ', fore_color=ui.BLACK, back_color=ui.GREEN, bold=True)
            + ' Training assistant finished sucessfully.'
        )
        print('         This is your model file: \'{}\''.format(model_file_name))
        self.path_to_sgdml_model = model_file_name
        model = SetupCalculator._load_model_sgdml(model_file_name)
        simulator.calculator.model = model



import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from source.dataset import Dataset
from source.metrics import db_eval_boundary, db_eval_iou
from source import utils
from source.results import Results
from scipy.optimize import linear_sum_assignment
from math import floor
import multiprocessing as mp
from multiprocessing import Manager


class Evaluation(object):
    def __init__(self, dataset_root, gt_set, sequences='all'):
        """
        Class to evaluate sequences from a certain set
        :param dataset_root: Path to the dataset folder that contains JPEGImages, Annotations, etc. folders.
        :param gt_set: Set to compute the evaluation
        :param sequences: Sequences to consider for the evaluation, 'all' to use all the sequences in a set.
        """
        self.dataset_root = dataset_root
        print(f"Evaluate on dataset = {self.dataset_root}")
        self.dataset = Dataset(root=dataset_root, subset=gt_set, sequences=sequences)

    @staticmethod
    def _evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
        if all_res_masks.shape[0] > all_gt_masks.shape[0]:
            print("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
            all_res_masks = all_res_masks[:all_gt_masks.shape[0]]
            # sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res = np.zeros(all_gt_masks.shape[:2])
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        return j_metrics_res

    def evaluate(self, res_path, metric=('J', 'J_last'), debug=False):
        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]

        # Containers
        metrics_res = {}
        if 'J' in metric:
            metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        if 'J_last' in metric:
            metrics_res['J_last'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

        process_list = []
        sema = mp.Semaphore(8)
        manager = Manager()
        metrics_res = manager.dict({
            'J': manager.dict(
                {
                    "M": manager.list(),
                    "R": manager.list(),
                    "D": manager.list(),
                    "M_per_object": manager.dict(),
                }
            ),
            'J_last': manager.dict(
                {
                    "M": manager.list(),
                    "R": manager.list(),
                    "D": manager.list(),
                    "M_per_object": manager.dict(),
                }
            ),
        })
        # Sweep all sequences
        results = Results(root_dir=res_path)
        for seq in tqdm(list(self.dataset.get_sequences())):
        # sequences = list(results.get_sequences())
        # for seq in tqdm(sequences):
            def evaluate():
                print(f"\n{seq}")
                all_gt_masks, all_void_masks, all_masks_id = self.dataset.get_all_masks(seq, True)
                num_objects = all_gt_masks.shape[0]
                all_gt_masks, all_masks_id = all_gt_masks[:, 1:-1, :, :], all_masks_id[1:-1]
                num_eval_frames = len(all_masks_id)
                last_quarter_ind = int(floor(num_eval_frames * 0.75))
                all_res_masks = results.read_masks(seq, all_masks_id)
                j_metrics_res = self._evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
                for ii in range(all_gt_masks.shape[0]):
                    seq_name = f'{seq}_{ii+1}'
                    if 'J' in metric:
                        [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                        metrics_res['J']["M"].append(JM)
                        metrics_res['J']["R"].append(JR)
                        metrics_res['J']["D"].append(JD)
                        metrics_res['J']["M_per_object"][seq_name] = JM
                    if 'J_last' in metric:
                        [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii][last_quarter_ind:])
                        metrics_res['J_last']["M"].append(JM)
                        metrics_res['J_last']["R"].append(JR)
                        metrics_res['J_last']["D"].append(JD)
                        metrics_res['J_last']["M_per_object"][seq_name] = JM

                # Show progress
                if debug:
                    sys.stdout.write(seq + '\n')
                    sys.stdout.flush()
                sema.release()
                print(f"{seq} complete ! ")
            sema.acquire()
            p = mp.Process(target=evaluate, args=())
            p.start()
            process_list.append(p)
        for process in tqdm(process_list):
            process.join()
        return metrics_res

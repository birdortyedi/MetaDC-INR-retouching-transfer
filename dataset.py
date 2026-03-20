import os
from torch.utils.data import Dataset


class RetouchTrainingDataset(Dataset):
    def __init__(self, root_dir):
        """
        Dataloader for the Training set.
        """
        self.root_dir = root_dir
        self.train_dir = os.path.join(root_dir, 'Train')
        self.natural_dir = os.path.join(self.train_dir, 'natural')
        self.presets_dir = os.path.join(self.train_dir, 'Presets')
        
        # Get all images in natural/
        self.images = [f for f in os.listdir(self.natural_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.images.sort()
        
        # Get all presets
        self.presets = [d for d in os.listdir(self.presets_dir) if os.path.isdir(os.path.join(self.presets_dir, d))]
        self.presets.sort()
        
        # Flattened dataset: (Image, Preset)
        self.items = []
        for img in self.images:
            for preset in self.presets:
                input_path = os.path.join(self.natural_dir, img)
                target_path = os.path.join(self.presets_dir, preset, img)
                if os.path.exists(target_path):
                    self.items.append({
                        'input_path': input_path,
                        'target_path': target_path,
                        'preset': preset,
                        'image_name': img
                    })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class RetouchEvaluationDataset(Dataset):
    def __init__(self, root_dir):
        """
        Dataloader for the Benchmark set.
        """
        self.root_dir = root_dir
        self.benchmark_dir = os.path.join(root_dir, 'Benchmark')
        self.test_dir = os.path.join(self.benchmark_dir, 'Test')
        self.test_ref_dir = os.path.join(self.benchmark_dir, 'Test_References')
        self.ref_file = os.path.join(self.benchmark_dir, 'references_file.txt')
        
        # Load reference mapping: TestImageName -> ReferenceImageName
        self.ref_map = {}
        if os.path.exists(self.ref_file):
            with open(self.ref_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        self.ref_map[parts[0].strip()] = parts[1].strip()
        
        # Get list of presets in Test/Presets
        # Presets are directories in Benchmark/Test/Presets
        self.presets = []
        presets_dir = os.path.join(self.test_dir, 'Presets')
        if os.path.exists(presets_dir):
            self.presets = [d for d in os.listdir(presets_dir) if os.path.isdir(os.path.join(presets_dir, d))]
        self.presets.sort()

        # Get list of test images (Natural)
        self.test_images = []
        natural_dir = os.path.join(self.test_dir, 'natural')
        if os.path.exists(natural_dir):
            self.test_images = [f for f in os.listdir(natural_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.test_images.sort()

        # Create flattened list of tasks: (Preset, TestImage)
        self.tasks = []
        for preset in self.presets:
            for img_name in self.test_images:
                if img_name in self.ref_map:
                    ref_img_name = self.ref_map[img_name]
                    self.tasks.append({
                        'preset': preset,
                        'test_image_name': img_name,
                        'ref_image_name': ref_img_name
                    })

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        preset = task['preset']
        test_img_name = task['test_image_name']
        ref_img_name = task['ref_image_name']

        # Paths
        # Input: Benchmark/Test/natural/test_img.jpg
        input_path = os.path.join(self.test_dir, 'natural', test_img_name)
        
        # Reference Input: Benchmark/Test_References/natural/ref_img.jpg
        ref_input_path = os.path.join(self.test_ref_dir, 'natural', ref_img_name)
        
        # Reference Output: Benchmark/Test_References/Presets/Preset_X/ref_img.jpg
        ref_output_path = os.path.join(self.test_ref_dir, 'Presets', preset, ref_img_name)

        # Ground Truth Path (for evaluation): Benchmark/Test/Presets/Preset_X/test_img.jpg
        gt_path = os.path.join(self.test_dir, 'Presets', preset, test_img_name)
        
        return {
            'input_path': input_path,
            'ref_input_path': ref_input_path,
            'ref_output_path': ref_output_path,
            'gt_path': gt_path,
            'preset': preset,
            'image_name': test_img_name
        }


class CompetitionDataset(Dataset):
    def __init__(self, root_dir):
        """
        Dataloader for the Competition dataset structure:
        sampleX/
            sampleX_input.jpg
            sampleX_before.jpg
            sampleX_after.jpg
        """
        self.root_dir = root_dir
        self.samples = []
        
        # Get all subdirectories
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        subdirs.sort()
        
        for d in subdirs:
            sample_dir = os.path.join(root_dir, d)
            files = os.listdir(sample_dir)
            
            # Identify files by keywords
            input_path = next((os.path.join(sample_dir, f) for f in files if '_input' in f), None)
            before_path = next((os.path.join(sample_dir, f) for f in files if '_before' in f), None)
            after_path = next((os.path.join(sample_dir, f) for f in files if '_after' in f), None)
            
            if input_path and before_path and after_path:
                self.samples.append({
                    'input_path': input_path,
                    'ref_input_path': before_path,
                    'ref_output_path': after_path,
                    'image_name': d # e.g., 'sample1'
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class RetouchDataset(Dataset):
    def __init__(self, root_dir):
        """
        Aggregates available supervised pairs (Input, GT) from standard directories.
        """
        self.items = []
        
        # Standard structure (Train, Benchmark/Test, Validation)
        search_paths = [
            os.path.join(root_dir, 'Train'),
            os.path.join(root_dir, 'Benchmark', 'Test'),
            os.path.join(root_dir, 'Validation')
        ]
        
        for base in search_paths:
            if not os.path.exists(base): continue
            
            nat_dir = os.path.join(base, 'natural')
            pre_dir = os.path.join(base, 'Presets')
            
            if os.path.exists(nat_dir) and os.path.exists(pre_dir):
                images = [f for f in os.listdir(nat_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                presets = [d for d in os.listdir(pre_dir) if os.path.isdir(os.path.join(pre_dir, d))]
                
                for img in images:
                    for preset in presets:
                        target_path = os.path.join(pre_dir, preset, img)
                        if os.path.exists(target_path):
                            self.items.append({
                                'input_path': os.path.join(nat_dir, img),
                                'target_path': target_path
                            })
        
        print(f"RetouchDataset initialized with {len(self.items)} total tasks.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

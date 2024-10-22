import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.ops 
from torchvision.tv_tensors import BoundingBoxes, Mask
import skimage.io
import skimage.color

class CocoDataLoader(Dataset):
    def __init__(self, coco, dataset_dir, subset, dataset, transform=None):
        self.coco = coco
        self.subset = subset
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.image_ids = list(dataset.image_ids)
        self.dataset = dataset
        self.class_info = dataset.class_info
        self.source_class_ids = {}

        # Adicione isso em seu __init__ para construir o mapeamento
        self._build_class_id_map()
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Recupere o ID da imagem baseado no índice
        image_id = self.image_ids[idx]
        # Carregue a imagem e os targets associados usando uma função dedicada
        image, targets = self._load_image_and_target(image_id)
        
        # Aplicar transformações na imagem e nos targets, se fornecido
        if self.transform is not None:
            try:
                image, targets = self.transform(image, targets)
            except Exception as err:
                print(err)
                print(idx)
                print(targets)
                print(targets["masks"].shape)
        return image, targets
    
    def _build_class_id_map(self):
        """Constrói um mapeamento de IDs de classe para índices."""
        self.id_to_index = {class_info['id']: index for index, class_info in enumerate(self.class_info)}

    def get_class_index(self, class_id):
        """Retorna o índice para um dado ID de classe."""
        return self.id_to_index.get(class_id, None)
    
    def _load_image_and_target(self, image_id):
        # Caminho da imagem
        try:
            image_info = self.dataset.image_info[image_id-1]
        except Exception as err:
            print(err)
            print(image_id)
            print
        image_path = image_info["path"]
        
        # Carregar a imagem
        image = Image.open(image_path).convert('RGB')
        
        # Carregar máscaras
        # ann_ids = self.coco.getAnnIds(imgIds=image_id)
        # annotations = self.coco.loadAnns(ann_ids)
        # masks = np.zeros((image.size[1], image.size[0], len(annotations)), dtype=np.uint8)
        mask1, class_ids = self.dataset.load_mask(image_id-1)
        # bbox = []
        # for i, ann in enumerate(annotations):
        #     # masks[:, :, i] = self.coco.annToMask(ann["segmentation"])
        #     bbox.append(ann["bbox"])
        mask1= torch.from_numpy(mask1)
        mask_tensor_bool = mask1.bool()
        mask_tensor= mask_tensor_bool.permute(2, 0, 1)
        if mask_tensor.size(0) > 0:  # Assuming `mask_tensor` is of shape [num_masks, height, width]
            # Gerar caixas delimitadoras a partir de máscaras
            try:
                boxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(mask_tensor), format='xyxy', canvas_size=image.size[::-1])

                #boxes = BoundingBoxes(data=bbox, format='xyxy',canvas_size=image.size[::-1])
            except Exception as err:
                print(err)
        else:
            # Handle the case for empty masks
            print("HERE")
            # For example, you could set `boxes` to an empty tensor of the appropriate shape
            boxes = torch.zeros((0, 4), dtype=torch.float32)


        # Etiquetas (IDs de classe)
        #labels = torch.tensor([self.get_class_index(ann['category_id']) for ann in annotations], dtype=torch.int64)
        #labels = class_ids
        targets = {'masks': Mask(mask_tensor), 'boxes': boxes, 'labels': torch.tensor(class_ids, dtype=torch.int64)}
        return TF.to_tensor(image), targets
    
    def _extract_bbox(self, mask):
        positions = np.where(mask)
        xmin = np.min(positions[1])
        xmax = np.max(positions[1])
        ymin = np.min(positions[0])
        ymax = np.max(positions[0])
        return [xmin, ymin, xmax, ymax]
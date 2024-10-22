import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.tv_tensors import BoundingBoxes, Mask
import skimage.io
import skimage.color

class CocoDataLoader(Dataset):
    def __init__(self, coco, dataset_dir, subset, class_info, transform=None):
        self.coco = coco
        self.subset = subset
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.image_ids = list(self.coco.imgs.keys())
        self.class_info = class_info
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
            image, targets = self.transform(image, targets)
        
        return image, targets
    
    def _build_class_id_map(self):
        """Constrói um mapeamento de IDs de classe para índices."""
        self.id_to_index = {class_info['id']: index for index, class_info in enumerate(self.class_info)}

    def get_class_index(self, class_id):
        """Retorna o índice para um dado ID de classe."""
        return self.id_to_index.get(class_id, None)
    
    def _load_image_and_target(self, image_id):
        # Caminho da imagem
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = f"{self.dataset_dir}/{self.subset}/images/{image_info['file_name']}"
        
        # Carregar a imagem
        image = Image.open(image_path).convert('RGB')
        
        # Carregar máscaras
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        masks = np.zeros((image.size[1], image.size[0], len(annotations)), dtype=np.uint8)
        for i, ann in enumerate(annotations):
            masks[:, :, i] = self.coco.annToMask(ann)
        
        # Converte a máscara numpy para tensor PyTorch
        mask_tensor = torch.tensor(masks, dtype=bool).permute(2, 0, 1)
        
        # Gerar caixas delimitadoras a partir de máscaras
        boxes = [self._extract_bbox(mask) for mask in masks.transpose(2, 0, 1)]
        boxes = BoundingBoxes(torch.as_tensor(boxes, dtype=torch.float32), format='xyxy', canvas_size=image.size[::-1])
        
        # Etiquetas (IDs de classe)
        labels = torch.tensor([self.get_class_index(ann['category_id']) for ann in annotations], dtype=torch.int64)
        
        targets = {'masks': Mask(mask_tensor), 'boxes': boxes, 'labels': labels}
        
        return TF.to_tensor(image), targets
    
    def _extract_bbox(self, mask):
        positions = np.where(mask)
        xmin = np.min(positions[1])
        xmax = np.max(positions[1])
        ymin = np.min(positions[0])
        ymax = np.max(positions[0])
        return [xmin, ymin, xmax, ymax]
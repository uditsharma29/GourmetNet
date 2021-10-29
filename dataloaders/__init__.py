from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, uecfoodpix, unimib
from torch.utils.data import DataLoader

from torchvision import transforms
import PIL

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
        
    elif args.dataset == 'foodpix':
        img_size = (args.base_size, args.base_size)
        num_class = 103
        train_transform = transforms.Compose([
		#transforms.RandomHorizontalFlip(),
        transforms.Resize(img_size, interpolation = PIL.Image.NEAREST),
        #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.ToTensor()
        #transforms.ToPILImage()
        ])
        val_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation = PIL.Image.NEAREST),
        #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.ToTensor()
    ])
        train_set = uecfoodpix.UECFoodPix(folder_path="/home/us2848/UECFoodPix/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/train", transforms = train_transform)
        val_set = uecfoodpix.UECFoodPix(folder_path="/home/us2848/UECFoodPix/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/test", transforms = val_transform)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False)
        test_loader = None
        
        return train_loader, val_loader, test_loader, num_class
        
    elif args.dataset == 'unimib':
        img_size = (args.width, args.base_size)
        num_class = 74
        train_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation = PIL.Image.NEAREST),
        #transforms.RandomHorizontalFlip(),
        #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.ToTensor(),
        
        #transforms.ToPILImage()
        ])
        val_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation = PIL.Image.NEAREST),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
        train_set = unimib.UNIMIB(folder_path="/home/us2848/UNIMIB dataset/train/", transforms = train_transform)
        val_set = unimib.UNIMIB(folder_path="/home/us2848/UNIMIB dataset/test/", transforms = val_transform)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False)
        test_loader = None
        
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError


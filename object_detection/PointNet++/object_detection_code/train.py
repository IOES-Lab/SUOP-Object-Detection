import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import PointNet2Detection
from dataset import DetectionDataset

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DetectionDataset(args.h5_train, num_points=args.num_points, train_mode=True)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    model = PointNet2Detection(num_classes=args.num_classes, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    cls_criterion = nn.CrossEntropyLoss()
    bbox_criterion = nn.SmoothL1Loss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for pts, bboxes, labels in train_loader:
            pts, bboxes, labels = pts.to(device), bboxes.to(device), labels.to(device)

            optimizer.zero_grad()
            pred_bbox, logits = model(pts)

            loss_cls = cls_criterion(logits, labels)
            loss_bbox = bbox_criterion(pred_bbox, bboxes)
            loss = 0.5 * loss_cls + 1.5 * loss_bbox

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            print(f"[Epoch {epoch}] Cls Loss: {loss_cls.item():.4f}, BBox Loss: {loss_bbox.item():.4f}, Total: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs} - Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"save: {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_train', type=str, required=True, help='tain HDF5 channel')
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--save_path', type=str, default='detector.pth')
    args = parser.parse_args()
    train(args)

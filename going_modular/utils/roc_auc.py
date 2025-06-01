import torch
import numpy as np
from sklearn.metrics import roc_auc_score

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

from ..model.modeling_output import ConcatMTLFaceRecognitionV3Outputs
from ..model.ConcatMTLFaceRecognition import ConcatMTLFaceRecognitionV3

def _get_predict_label_class(similarity_matrix: torch.Tensor, all_ids: torch.Tensor, num_classes:int):
    similarity_matrix = torch.diagonal_scatter(similarity_matrix, torch.tensor([-1000]*similarity_matrix.shape[0])).cpu()
    predict_class_index = all_ids[similarity_matrix.argmax(dim = -1)].unsqueeze(1).cpu().to(torch.int64)

    predict_class = torch.zeros((predict_class_index.shape[0], num_classes), dtype= torch.int8,device = 'cpu')
    predict_class = predict_class.scatter_(
        dim=1,
        index=predict_class_index,
        src=torch.ones_like(predict_class_index, dtype= predict_class.dtype, device = 'cpu')
    )

    label_class = torch.zeros((all_ids.shape[0], num_classes),dtype= torch.int8, device = 'cpu')
    label_class = label_class.scatter_(
        dim=1, 
        index=all_ids.unsqueeze(1).cpu().to(torch.int64),
        src=torch.ones_like(all_ids.unsqueeze(1), dtype= label_class.dtype, device = 'cpu')
    )

    return predict_class, label_class

def compute_auc(
    dataloader: torch.utils.data.DataLoader, 
    model: ConcatMTLFaceRecognitionV3, 
    device: str,
    num_classes:int
):
    
    model.eval()
    all_labels = {'gender': [], 'spectacles': [], 'facial_hair': [], 'pose': [], 'emotion': [] }
    all_preds = {'gender': [], 'spectacles': [], 'facial_hair': [], 'pose': [], 'emotion': [] }

    with torch.no_grad():
        embeddings_list = []

        for batch in dataloader:
            images, y = batch
            # Lấy các nhãn thực tế từ y
            id, gender, spectacles, facial_hair, pose, emotion = (
                y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5]
            )
            
            images = images.to(device)
            # Trừ id, còn lại đều đã qua softmax
            outputs = model(images)

            x_id = outputs.id_embedding
            x_spectacles, x_facial_hair, x_pose, x_emotion, x_gender, _  = outputs.logits.to_tuple()

            # Append IDs and embeddings
            embeddings_list.append((id, x_id))

            # Tính xác suất và lớp dự đoán cho mỗi thuộc tính (gender, spectacles, facial_hair, pose, occlusion, emotion)
            for attribute, x_attr, y_attr in zip(
                ['gender', 'spectacles', 'facial_hair', 'pose', 'emotion'],
                [torch.softmax(x_gender, dim=1), torch.softmax(x_spectacles, dim=1), torch.softmax(x_facial_hair, dim=1), 
                torch.softmax(x_pose, dim=1), torch.softmax(x_emotion, dim=1)],
                [gender, spectacles, facial_hair, pose, emotion]
            ):
                # Tìm xác suất lớn nhất và lớp của nó
                probs, predicted_classes = torch.max(x_attr, dim=1)

                # Đảm bảo y_attr cùng device với predicted_classes
                y_attr = y_attr.to(predicted_classes.device)

                # Cập nhật all_labels với việc so sánh lớp dự đoán với nhãn thực tế
                all_labels[attribute].append((predicted_classes == y_attr).int().cpu().numpy())

                # Lấy xác suất cho lớp dự đoán đúng từ x_attr
                all_preds[attribute].append(torch.gather(x_attr, 1, y_attr.unsqueeze(1).long()).squeeze(1).cpu().numpy())

        # Chuyển thành mảng 1 chiều cho tất cả các lớp
        for attribute in all_labels:
            all_labels[attribute] = np.concatenate(all_labels[attribute], axis=0)
        for attribute in all_preds:
            all_preds[attribute] = np.concatenate(all_preds[attribute], axis=0)


        # Concatenate all id embeddings into one tensor
        all_ids = torch.cat([x[0] for x in embeddings_list], dim=0)
        all_embeddings = torch.cat([x[1] for x in embeddings_list], dim=0)

        # print("all_ids: ",all_ids.shape)
        # print("all_embeddings: ",all_embeddings.shape)
        # print('num_classes: ', num_classes)
        
        euclidean_scores = []
        euclidean_labels = []
        cosine_scores = []
        cosine_labels = []

        # Compute pairwise Euclidean distance and cosine similarity
        all_embeddings_norm = all_embeddings / all_embeddings.norm(p=2, dim=1, keepdim=True)
        euclidean_distances = -torch.cdist(all_embeddings, all_embeddings, p=2)  # Euclidean distance matrix
        cosine_similarities = torch.mm(all_embeddings_norm, all_embeddings_norm.t())  # Cosine similarity matrix
        
        # Compute labels (same id = 0, different id = 1)
        # labels = (all_ids.unsqueeze(1) == all_ids.unsqueeze(0)).int().to(device)

        # Flatten and filter results
        # euclidean_scores = euclidean_distances[torch.triu(torch.ones_like(labels), diagonal=1) == 1].cpu().numpy()
        # euclidean_labels = labels[torch.triu(torch.ones_like(labels), diagonal=1) == 1].cpu().numpy()
        
        # cosine_scores = cosine_similarities[torch.triu(torch.ones_like(labels), diagonal=1) == 1].cpu().numpy()
        # cosine_labels = labels[torch.triu(torch.ones_like(labels), diagonal=1) == 1].cpu().numpy()

        euclidean_predict_class, euclidean_label_class = _get_predict_label_class(
            similarity_matrix= euclidean_distances, 
            all_ids= all_ids, 
            num_classes= num_classes
        )

        cosine_predict_class, cosin_label_class = _get_predict_label_class(
            similarity_matrix= cosine_similarities, 
            all_ids= all_ids, 
            num_classes= num_classes
        )

        # Compute ROC AUC for Euclidean distance
        all_labels['id_euclidean'] = euclidean_label_class.numpy()
        all_preds['id_euclidean'] = euclidean_predict_class.numpy()

        # Compute ROC AUC for Cosine similarity
        all_labels['id_cosine'] =  cosin_label_class.numpy()     # np.array(cosine_labels)
        all_preds['id_cosine'] = cosine_predict_class.numpy()       #np.array(cosine_scores)
        # print('check shape: ', label_class.shape, predict_class.shape)
        
        # # Calculate accuracy for Euclidean distance
        # euclidean_optimal_idx = np.argmax(tpr_euclidean - fpr_euclidean) # Chọn ngưỡng tại điểm có giá trị tpr - fpr lớn nhất trên đường ROC, vì đây là nơi tối ưu hóa sự cân bằng giữa tỷ lệ phát hiện (TPR) và tỷ lệ báo động giả (FPR).
        # euclidean_optimal_threshold = thresholds_euclidean[euclidean_optimal_idx]
        # euclidean_pred_labels = (euclidean_pred_scores >= euclidean_optimal_threshold).astype(int)
        # euclidean_accuracy = accuracy_score(euclidean_true_labels, euclidean_pred_labels)

        # # Calculate accuracy for Cosine similarity
        # cosine_optimal_idx = np.argmax(tpr_cosine - fpr_cosine)
        # cosine_optimal_threshold = thresholds_cosine[cosine_optimal_idx]
        # cosine_pred_labels = (cosine_pred_scores >= cosine_optimal_threshold).astype(int)
        # cosine_accuracy = accuracy_score(cosine_true_labels, cosine_pred_labels)
        
        # Tính AUC cho từng tác vụ
        auc_scores = {}
        for task in all_labels:
            unique_classes = np.unique(all_labels[task])
            if len(unique_classes) < 2:
                # Nếu chỉ có một lớp, gán AUC là None hoặc giá trị mặc định
                auc_scores[task] = 1
            else:
                auc_scores[task] = roc_auc_score(all_labels[task], all_preds[task])
                
        return auc_scores



# def auc_two_dataloder(
#         test_dataloader: torch.utils.data.DataLoader, 
#         gallery_dataloader: torch.utils.data.DataLoader, 
#         model: ConcatMTLFaceRecognitionV3, 
#         device: str,
#         num_classes:int
#     ):

#     # build storage
#     embeddings_storage = []
#     with torch.no_grad():
#         for batch in test_dataloader:
#             images, y = batch
#             # Lấy các nhãn thực tế từ y
#             id = y[:, 0]

#             outputs = model(images)
#             output_embedding = outputs.id_embedding
#             embeddings_storage.append((id, output_embedding))
    

import pathlib, cv2
import numpy as np
from data import data, coordinates, heatmap_set, visibility

def draw_images(model, img_idxs: list[int]):  
    pathlib.Path("result").mkdir(parents=True, exist_ok=True)
    # GENERATE RESULT IMAGES
    filenames = []
    for t in img_idxs:
        _, preds, _ = model.predict(data[t:t+1])
        gt_skeleton = coordinates[t].astype(np.uint8)
        pred_skeleton = preds[0].astype(np.uint8)
        assert gt_skeleton.shape == pred_skeleton.shape
        img = data[t].astype(np.uint8)
        # draw the joints
        for i in range(14):
            cv2.circle(img, center=tuple(gt_skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=2)
            cv2.circle(img, center=tuple(pred_skeleton[i][0:2]), radius=2, color=(0,0,255), thickness=2)
        # draw the lines
        for j in ((13, 12), (12, 8), (12, 9), (8, 7), (7, 6), (9, 10), (10, 11), (2, 3), (2, 1), (1, 0), (3, 4), (4, 5)):
            cv2.line(img, tuple(gt_skeleton[j[0]][0:2]), tuple(gt_skeleton[j[1]][0:2]), color=(0, 255, 0), thickness=1)
            cv2.line(img, tuple(pred_skeleton[j[0]][0:2]), tuple(pred_skeleton[j[1]][0:2]), color=(0, 0, 255), thickness=1)
        # solve the mid point of the hips
        cv2.line(img, tuple(gt_skeleton[12][0:2]), tuple(gt_skeleton[2][0:2] // 2 + gt_skeleton[3][0:2] // 2), color=(0, 255, 0), thickness=1)
        cv2.line(img, tuple(pred_skeleton[12][0:2]), tuple(pred_skeleton[2][0:2] // 2 + pred_skeleton[3][0:2] // 2), color=(0, 0, 255), thickness=1)
        cv2.putText(img, 'green = gt', (130,230), 1, 0.8, color=(0,255,0), thickness=1)
        cv2.putText(img, 'red = predicted', (130,245), 1, 0.8, color=(0,0,255), thickness=1)

        pck_score = model.evaluate(data[t:t+1], [heatmap_set[t:t+1], coordinates[t:t+1], visibility[t:t+1]])[-1]
        cv2.putText(img, f'pck score: {pck_score:.3f}', (10,15), 1, 0.8, color=(0,0,0), thickness=1)

        filename = "./result/lsp_%d.jpg"%t
        cv2.imwrite(filename, img)
        filenames.append(filename)
    return filenames
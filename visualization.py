import pathlib, cv2
import numpy as np
from data import data, coordinates, visibility

keypoints_dict = {
    0: "Right ankle",
    1: "Right knee",
    2: "Right hip",
    3: "Left hip",
    4: "Left knee",
    5: "Left ankle",
    6: "Right wrist",
    7: "Right elbow",
    8: "Right shoulder",
    9: "Left shoulder",
    10: "Left elbow",
    11: "Left wrist",
    12: "Neck",
    13: "Head top",
}


def draw_images_without_predictions(img_idxs: list[int]):
    pathlib.Path("result").mkdir(parents=True, exist_ok=True)
    # GENERATE RESULT IMAGES
    filenames = []
    for t in img_idxs:
        gt_skeleton = coordinates[t].astype(np.uint8)
        img = data[t].astype(np.uint8)

        # draw the joints
        for i in range(14):
            cv2.circle(
                img,
                center=tuple(gt_skeleton[i][0:2]),
                radius=2,
                color=(0, 255, 0),
                thickness=2,
            )
            print(f"keypoint {keypoints_dict[i]} for idx {t}: {gt_skeleton[i][0:2]}")

        # draw the lines
        for j in (
            (13, 12),
            (12, 8),
            (12, 9),
            (8, 7),
            (7, 6),
            (9, 10),
            (10, 11),
            (2, 3),
            (2, 1),
            (1, 0),
            (3, 4),
            (4, 5),
        ):
            cv2.line(
                img,
                tuple(gt_skeleton[j[0]][0:2]),
                tuple(gt_skeleton[j[1]][0:2]),
                color=(0, 255, 0),
                thickness=1,
            )

        # solve the mid point of the hips
        cv2.line(
            img,
            tuple(gt_skeleton[12][0:2]),
            tuple(gt_skeleton[2][0:2] // 2 + gt_skeleton[3][0:2] // 2),
            color=(0, 255, 0),
            thickness=1,
        )

        cv2.putText(
            img, "green = gt", (130, 230), 1, 0.8, color=(0, 255, 0), thickness=1
        )

        filename = "./result/lspet_%d.jpg" % t
        print(filename)
        cv2.imwrite(filename, img)
        filenames.append(filename)
    return filenames


# draw_images_without_predictions([2])
print(visibility)

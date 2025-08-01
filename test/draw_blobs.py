import torch, numpy as np, matplotlib.pyplot as plt

if __name__ == '__main__':
    points = torch.load('random_points.zip').numpy()
    blobs = torch.load('solved_blobs.zip').numpy()
    views = torch.load('views.zip').numpy()
    print('Views')
    print(views)
    #Scatter the points
    n=1000
    plt.scatter(points[:n, :,0], points[:n, :,1], s=.5)

    dirs = views[:, 1] - views[:, 0]
    print('Dirs')
    print(dirs)

    # Define the 90-degree counter-clockwise rotation matrix
    rotation_matrix_ccw = np.array(
        [[0, -1],
        [1,  0]]
    )
    # rot = np.roll(dirs, 1, axis=1)
    rot = np.zeros_like(dirs)
    for i in range(dirs.shape[0]): rot[i] = np.dot(rotation_matrix_ccw, dirs[i])

    print(rot)

    colors = ['k', 'grey', 'r', 'c', 'm']

    #iterate over the blobs
    print(f'Found {blobs.shape[0]} blobs')
    for i in range(min([10, blobs.shape[0]])):
        blob = blobs[i]
        # print('Blob\n', blob)
        blob_starts = views[:,0] + dirs*blob[:,0].reshape(-1, 1)
        blob_ends = views[:,0] + dirs*blob[:,1].reshape(-1, 1)
        # print('Blob starts\n', blob_starts)
        # print('Blob ends\n', blob_ends)

        for j in range(blob.shape[0]):
            plt.axline(blob_starts[j], blob_starts[j]+rot[j], c=colors[j])
            plt.axline(blob_ends[j], blob_ends[j]+rot[j], c=colors[j])

    plt.xlim(-50, 4050)
    plt.ylim(-50, 4050)
    plt.show()
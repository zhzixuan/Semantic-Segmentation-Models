import os


path = '/home/user/Program/caffe-model/seg/FinalProject/ssp_513_124/demo_eval'

#
# def listdir(path, list_name):
#     for file in os.listdir(path):
#         file_path = os.path.join(path, file)
#         if os.path.isdir(file_path):
#             listdir(file_path, list_name)
#         elif os.path.splitext(file_path)[1] == '.jpg':
#             list_name.append(file_path)

def file_name(file_dir):
    L = []
    f = open("/home/user/Program/caffe-model/seg/FinalProject/ssp_513_124/demo_eval.txt", "w")
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                # L.append(os.path.join(root, file))
                L.append(file)
                f.write(file + '\n')
                f.close
    return L

if __name__ == '__main__':
    demolist = file_name(path)

    # print demolist

import subprocess

def run(command):
    subprocess.check_call("python {}".format(command), shell=True)

if __name__ == "__main__":
    """
    mkdir ../model
    mkdir ../model/feature
    mkdir ../model/model
    mkdir ../model/pred
    mkdir ../submission

    run("preprocess_index.py")
    preprocess_index.py
    preprocess_img_to_hdf5.py
    preprocess_img_to_hdf5_64_raw.py

    run_keras_cnn4.py
    run_keras_cnn5.py
    run_keras_cnn5_2.py
    run_keras_cnn5_3.py
    run_keras_cnn5_4.py
    run_keras_resnet.py
    run_keras_resnet_augmenation.py
    run_keras_vgg2.py
    run_keras_vgg2_augmentation.py

    run_mix2.py
    run_mix_greedy_fin.py
    run_keras_mlp_fin.py
    run_mix_greedy_fin2.py
    run_adjuster.py
    """
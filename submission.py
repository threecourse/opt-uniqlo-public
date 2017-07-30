import numpy as np
import pandas as pd
import util

class Submission:

    @classmethod
    def make_submission(cls, run_name):
        pred_path = "../model/pred/pred_values_{}_test.csv".format(run_name)
        pred_test = pd.read_csv(pred_path, sep="\t")
        pred_test = pred_test.values

        # TODO use decode
        df = pd.read_csv("../model/filenames_index_train.csv", sep="\t")
        ratio = df["cid"].value_counts()
        ratio = ratio / float(len(df))

        for c in range(24):
            pred_test[:, c] /= ratio[c]

        pred_label = np.argmax(pred_test, axis=1)

        df_test = pd.read_csv("../model/filenames_index_test.csv", sep="\t")
        submission = df_test[["file_name"]].copy()
        submission["cid"] = pred_label
        submission.to_csv("../submission/subm_{}_{}.csv".format(util.Util.nowstr(), run_name),
                          sep=",", index=False, header=None)

if __name__ == "__main__":
    # Submission.make_submission("../model/pred_values_xgb_stack1_test.csv", "xgb_stack1")

    # Submission.make_submission("../model/pred/pred_values_xgb4_test.csv", "xgb4_again")
    # Submission.make_submission("../model/pred/pred_values_xgb_stack1_test.csv", "xgb_stack1_again")
    # Submission.make_submission("../model/pred/pred_values_keras_mlp_stack1_test.csv", "keras_mlp_stack1")

    # Submission.make_submission("../model/pred/pred_values_vgg1_test.csv", "vgg1")
    # Submission.make_submission("../model/pred/pred_values_vgg1_e5_test.csv", "vgg1_e5")
    # Submission.make_submission("../model/pred/pred_values_vgg1_e7_test.csv", "vgg1_e7")
    # Submission.make_submission("../model/pred/pred_values_vgg1_e6_test.csv", "vgg1_e6")
    # Submission.make_submission("../model/pred/pred_values_vgg1_e8_test.csv", "vgg1_e8")
    # Submission.make_submission("../model/pred/pred_values_keras_mlp_stack1_test.csv", "keras_mlp_stack1")
    # Submission.make_submission("vgg1_mix5")
    """
    Submission.make_submission("vgg2_mix")
    Submission.make_submission("vgg2_mix2")
    Submission.make_submission("vgg2_mix3")
    Submission.make_submission("vgg2_mix4")
    Submission.make_submission("vgg2_mix5")

    Submission.make_submission("vgg2_mix6")
    Submission.make_submission("vgg2_mix7")
    Submission.make_submission("vgg2_mix8")
    Submission.make_submission("vgg2_mix9")
    Submission.make_submission("vgg2_mix10")

    Submission.make_submission("resnet_e19")
    Submission.make_submission("resnet_mix2")
    Submission.make_submission("mixes_a")
    Submission.make_submission("mixes_b")

    Submission.make_submission("mixes_c")
    Submission.make_submission("resnet_mix10")

    Submission.make_submission("inception_e21")
    Submission.make_submission("inception_mix5")
    Submission.make_submission("mixes_d")
    Submission.make_submission("mix_colors1")
    Submission.make_submission("resnet_multi_e19")
    Submission.make_submission("keras_cnn4")
    Submission.make_submission("cnn_mix_15")
    Submission.make_submission("mix_x5")
    Submission.make_submission("resnet_aug_e25")
    Submission.make_submission("resnet_aug_mix4")
    Submission.make_submission("mix_xx5")
    Submission.make_submission("resnet_aug_mix9")
    Submission.make_submission("keras_cnn5_e73")
    Submission.make_submission("mix_xx7")
    """

    Submission.make_submission("vgg2_aug_temp_checkepoch")





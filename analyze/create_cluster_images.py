import sys, os
sys.path.append("")

from util import Util
import numpy as np
import pandas as pd
from skimage import io as skio
import matplotlib.pyplot as plt
import skimage
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as PathEffects

class ClusterImagesCreator:

    @classmethod
    def run(cls, out_path, _df, n_clusters=10, folder="../data/train/", cid_actual=True):

        df_colors = pd.read_csv("a00_analyze/color_info.csv")

        assert("file_name" in _df.columns)
        assert("cls" in _df.columns)
        assert("cid" in _df.columns)

        with PdfPages(out_path) as pdf:
            for c in range(n_clusters):
                print "generating cluster {}".format(c)

                df = _df[_df["cls"]==c]
                df = df.sort_values("cid")
                img_h, img_w = 100, 100
                ary_h, ary_w = 1000, 1500
                H, W = ary_h / img_h, ary_w / img_w

                for ii, (_, r) in enumerate(df.iterrows()):

                    if ii % (H * W) == 0:
                        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
                        aryl = np.zeros((ary_h, ary_w, 3))
                        page_no = (ii / (H * W)) + 1

                    fname = r["file_name"]
                    path = os.path.join(folder, fname)
                    img = skio.imread(path)
                    img = skimage.transform.resize(img, (img_h, img_w), mode='reflect')

                    label = r["cid"]
                    code = df_colors.iloc[label]["code"]
                    color1 = df_colors.iloc[label]["color1"]

                    iii = ii % (H * W)
                    x = iii % W
                    y = iii / W
                    x1, x2 = x * img_w, x * img_w + img_w
                    y1, y2 = y * img_h, (y + 1) * img_h
                    aryl[y1:y2, x1:x2, :] = np.array(img)
                    if cid_actual:
                        txt = plt.text(x1, y1+15, code, color=color1, fontsize=12)
                        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='gray')])
                    else:
                        txt = plt.text(x1, y1 + 15, code, color=color1, fontsize=12)
                        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='pink')])
                    plt.draw()

                    if (ii+1) % (H * W) == 0 or (ii+1) == len(df):
                        plt.title("cls{} - {}".format(c, page_no), fontsize=48)
                        plt.imshow(aryl)
                        pdf.savefig()
                        plt.close()

if __name__ == "__main__":
    pass

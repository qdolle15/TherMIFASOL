import os
import numpy as np

if __name__ == "__main__":

    path = "./data/experiences/BoardCardFilter/DEC4_mat/CRED/"
    list_of_layer = np.arange(0, 28)

    for layer in list_of_layer:
        layer =29
        sub_path = os.path.join(path, f"data/layer_{layer+1}")
        os.mkdir(sub_path)

        full_array = np.load(os.path.join(path, f"cordon_{layer}_0.npy"))
        for cpt, im in enumerate(full_array):
            np.save(os.path.join(sub_path, f"im_{cpt}.npy"), im)

        if os.path.exists(os.path.join(path, f"cordon_{layer}_1.npy")):
            full_array = np.load(os.path.join(path, f"cordon_{layer}_1.npy"))
            for cpt_, im in enumerate(full_array):
                np.save(os.path.join(sub_path, f"im_{cpt_ + cpt +1}.npy"), im)

        if os.path.exists(os.path.join(path, f"cordon_{layer}_2.npy")):
            full_array = np.load(os.path.join(path, f"cordon_{layer}_2.npy"))
            for cpt__, im in enumerate(full_array):
                np.save(os.path.join(sub_path, f"im_{cpt__ + cpt_ + cpt +1}.npy"), im)

        if os.path.exists(os.path.join(path, f"cordon_{layer}_3.npy")):
            full_array = np.load(os.path.join(path, f"cordon_{layer}_3.npy"))
            for cpt___, im in enumerate(full_array):
                np.save(os.path.join(sub_path, f"im_{cpt___ + cpt__ + cpt_ + cpt +1}.npy"), im)
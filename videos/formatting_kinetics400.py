# %% Format the downloaded kinetics400 to the mmaction2 style.
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# (Optional). Rename class name to be only filled with characters from [_, (, ), ']. Original class names use [_, -]
def rename_classnames(path):
    folder1 = path / "hurling_-sport-"
    folder1.rename(folder1.with_stem("hurling_(sport)"))
    folder2 = path / "passing_American_football_-not_in_game-"
    folder2.rename(folder2.with_stem("passing_American_football_(not_in_game)"))
    folder3 = path / "skiing_-not_slalom_or_crosscountry-"
    folder3.rename(folder3.with_stem("skiing_(not_slalom_or_crosscountry)"))
    folder4 = path / "punching_person_-boxing-"
    folder4.rename(folder4.with_stem("punching_person_(boxing)"))
    folder5 = path / "massaging_person-s_head"
    folder5.rename(folder5.with_stem("massaging_person's_head"))
    folder6 = path / "shooting_goal_-soccer-"
    folder6.rename(folder6.with_stem("shooting_goal_(soccer)"))
    folder7 = path / "petting_animal_-not_cat-"
    folder7.rename(folder7.with_stem("petting_animal_(not_cat)"))
    folder8 = path / "tying_knot_-not_on_a_tie-"
    folder8.rename(folder8.with_stem("tying_knot_(not_on_a_tie)"))
    folder9 = path / "using_remote_controller_-not_gaming-"
    folder9.rename(folder9.with_stem("using_remote_controller_(not_gaming)"))
    folder10 = path / "passing_American_football_-in_game-"
    folder10.rename(folder10.with_stem("passing_American_football_(in_game)"))


def get_real_id(id, ann, youtube_id, youtube_id_clean, with_suffix=True):
    id_clean = id.replace("_", "").replace("-", "")
    idx = youtube_id_clean.index(id_clean)
    real_id = youtube_id[idx]
    if len(real_id) != 11:
        raise ValueError
    if with_suffix:
        start = ann.iloc[idx].time_start
        end = ann.iloc[idx].time_end
        real_id = f"{real_id}_{start:06d}_{end:06d}"
    return real_id


def format_train():
    train_ann = pd.read_csv('/home/louis/PycharmProjects/APN/data/kinetics400/annotations/kinetics_train.csv')
    train_p = Path("/home/louis/PycharmProjects/APN/data/kinetics400/videos_train")
    youtube_id = train_ann.youtube_id.values
    youtube_id_clean = [id_ann.replace("_", "").replace("-", "") for id_ann in youtube_id]

    # rename_classnames(train_p)    # optional

    processed = 0
    for vp in tqdm(list(train_p.glob('**/*.*'))):
        if "youtube" in vp.stem:
            vp_id = vp.stem.split("-", 1)[-1]
            vp_id = vp_id.rsplit("-", 2)[0]
            real_vp_id = get_real_id(vp_id, train_ann, youtube_id, youtube_id_clean)
            vp.rename(vp.with_stem(real_vp_id))
            processed += 1
        else:
            vp_id = vp.stem.rsplit("_", 2)[0]
            if len(vp_id) == 11:
                continue
            else:
                real_vp_id = get_real_id(vp_id, train_ann, youtube_id, youtube_id_clean)
                vp.rename(vp.with_stem(real_vp_id))
                processed += 1
    print(f"Processed:\t{processed}")


def format_val():
    val_ann = pd.read_csv('/home/louis/PycharmProjects/APN/data/kinetics400/annotations/kinetics_val.csv')
    val_p = Path("/home/louis/PycharmProjects/APN/data/kinetics400/videos_val")

    youtube_id = val_ann.youtube_id.values
    youtube_id_clean = [id_ann.replace("_", "").replace("-", "") for id_ann in youtube_id]

    # rename_classnames(val_p)    # optional

    processed = 0
    for vp in tqdm(list(val_p.glob('**/*.*'))):
        if "youtube" in vp.stem:
            vp_id = vp.stem.split("-", 1)[-1]
            vp_id = vp_id.rsplit("-", 2)[0]
            real_vp_id = get_real_id(vp_id, val_ann, youtube_id, youtube_id_clean)
            vp.rename(vp.with_stem(real_vp_id))
            processed += 1
        else:
            vp_id = vp.stem.split('.', 1)[0]
            suffixes = vp.name.split('.', 1)[1:]
            real_vp_id = get_real_id(vp_id, val_ann, youtube_id, youtube_id_clean)
            vp.rename(vp.with_stem(real_vp_id))
            processed += 1
    print(f"Processed:\t{processed}")

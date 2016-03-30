__author__ = 'charles'
from db_marche.features.utils_step import *
import numpy as np
import logging
from itertools import product, cycle

fps=100
def feat_StepNum(e):
    """
    Number of steps per minute. number of HS / walking duration
    """
    res = dict()

    phase_dict = {"For": (e.seg_annotation[0], e.seg_annotation[1]),
                  "Bac": (e.seg_annotation[2], e.seg_annotation[3])}
    f = lambda x: np.array(x)
    foot_dict = {
        "Rig": (
            f(extract_right_to_from_exo(e)), f(extract_right_hs_from_exo(e)),
            f(extract_left_to_from_exo(e)), f(extract_left_hs_from_exo(e))),
        "Lef": (f(extract_left_to_from_exo(e)), f(extract_left_hs_from_exo(e)),
                f(extract_right_to_from_exo(e)),
                f(extract_right_hs_from_exo(e)))}
    for (phase, (start, end)), (
            foot,
            (current_to, current_hs, opposite_to, opposite_hs)) in product(
            phase_dict.items(), foot_dict.items()):
        tmp = (current_hs > start) * (current_hs < end)

        foot_string, phase_string = "left", "forward"
        if foot == "Rig": foot_string = "right"
        if phase == "Bac": phase_string = "back"

        walking_duration = (end - start + 1) / fps / 60  # in minutes
        walking_distance = e.meta["distance_parcourue_(m)"]
        step_per_min = len(tmp) / walking_duration
        step_per_meter = len(tmp) / walking_distance

        res[phase + foot + "StepPerMin"] = (
            step_per_min,
            "Number of " + foot_string + " heel strikes per minute during the "
                                         " walk " + phase_string + ".")

        res[phase + foot + "StepPerMeter"] = (
            step_per_meter,
            "Number of " + foot_string + " heel strikes per meter during the"
                                         " walk " + phase_string + ".")

    return res


def feat_StrideDuration(e):
    """
    Stride duration: heel strike to heel strike duration (same foot)
    """
    res = dict()
    f_start, f_end = e.seg_annotation[0], e.seg_annotation[1]
    b_start, b_end = e.seg_annotation[2], e.seg_annotation[3]

    # Right foot
    hs_times = np.array(extract_right_hs_from_exo(e))
    hs_times_forward = hs_times[(hs_times > f_start) * (hs_times < f_end)]
    hs_times_back = hs_times[(hs_times > b_start) * (hs_times < b_end)]
    strides_forward = np.diff(hs_times_forward) / fps
    strides_back = np.diff(hs_times_back) / fps

    if len(strides_forward) == 0:  # no step detected during this phase
        m = 0
        s = 0
        logging.warning("error for ForRigMeanStride and ForRigStdStride: "
                        + e.fname)
    else:
        m = np.mean(strides_forward)
        s = np.std(strides_forward)
    res["ForRigMeanStride"] = (m,
                               "Mean stride duration (heel strike to heel "
                               "strike) of the right foot during the walking "
                               "forward phase.")
    res["ForRigStdStride"] = (s,
                              "Standard deviation of the stride durations(heel "
                              "strike to heel strike) of the right footduring "
                              "the walking forward phase.")

    if len(strides_back) == 0:  # no step detected during this phase
        m = 0
        s = 0
        logging.warning("error for BacRigMeanStride and BacRigStdStride: "
                        + e.fname)
    else:
        m = np.mean(strides_back)
        s = np.std(strides_back)

    res["BacRigMeanStride"] = (m,
                               "Mean stride duration (heel strike to heel "
                               "strike) of the right foot during the walking "
                               "back phase.")
    res["BacRigStdStride"] = (s,
                              "Standard deviation of the stride durations "
                              "(heel strike to heel strike) of the right foot "
                              "during the walking back phase.")

    # Left foot
    hs_times = np.array(extract_left_hs_from_exo(e))
    hs_times_forward = hs_times[(hs_times > f_start) * (hs_times < f_end)]
    hs_times_back = hs_times[(hs_times > b_start) * (hs_times < b_end)]
    strides_forward = np.diff(hs_times_forward) / fps
    strides_back = np.diff(hs_times_back) / fps

    if len(strides_forward) == 0:  # no step detected during this phase
        m = 0
        s = 0
        logging.warning("error for ForLefMeanStride and ForLefStdStride: "
                        + e.fname)
    else:
        m = np.mean(strides_forward)
        s = np.std(strides_forward)
    res["ForLefMeanStride"] = (m,
                               "Mean stride duration (heel strike to heel "
                               "strike) of the left foot during the walking "
                               "forward phase.")
    res["ForLefStdStride"] = (s,
                              "Standard deviation of the stride durations(heel "
                              "strike to heel strike) of the left foot during "
                              "the walking forward phase.")

    if len(strides_back) == 0:  # no step detected during this phase
        m = 0
        s = 0
        logging.warning("error for BacLefMeanStride and BacLefStdStride: "
                        + e.fname)
    else:
        m = np.mean(strides_back)
        s = np.std(strides_back)

    res["BacLefMeanStride"] = (m,
                               "Mean stride duration (heel strike to heel "
                               "strike) of the left foot during the walking "
                               "back phase.")
    res["BacLefStdStride"] = (s,
                              "Standard deviation of the stride durations "
                              "(heel strike to heel strike) of the left foot "
                              "during the walking back phase.")

    return res


def feat_SwingPhase(e):
    """
    On one foot: (hs --> to)/(hs --> hs)
    """
    res = dict()
    f_start, f_end = e.seg_annotation[0], e.seg_annotation[1]
    b_start, b_end = e.seg_annotation[2], e.seg_annotation[3]

    # Right foot
    hs_times = np.array(extract_right_hs_from_exo(e))
    hs_times_forward = hs_times[(hs_times > f_start) * (hs_times < f_end)]
    hs_times_back = hs_times[(hs_times > b_start) * (hs_times < b_end)]

    to_times = np.array(extract_right_to_from_exo(e))
    to_times_forward = to_times[(hs_times > f_start) * (hs_times < f_end)]
    to_times_back = to_times[(hs_times > b_start) * (hs_times < b_end)]

    # right foot forward
    if len(hs_times_forward) < 2:  # one or no step detected during this phase
        m = 0
        s = 0
        logging.warning("error for ForRigMeanSwing and ForRigStdSwing: "
                        + e.fname)
    else:
        hs_to = to_times_forward[1:] - hs_times_forward[:-1]
        hs_hs = np.diff(hs_times_forward)
        m = np.mean(hs_to / hs_hs)
        s = np.std(hs_to / hs_hs)

    res["ForRigMeanSwing"] = (m,
                              "Mean ratio of the swing phase relative to the "
                              "stride, for the right foot during the walk "
                              "forward")
    res["ForRigStdSwing"] = (s,
                             "Standard deviation of the ratio of the swing "
                             "phase relative to the stride, for the right foot "
                             "during the walk forward")
    # right foot back
    if len(hs_times_back) < 2:  # one or no step detected during this phase
        m = 0
        s = 0
        logging.warning("error for BacRigMeanSwing and BacRigStdSwing: "
                        + e.fname)
    else:
        hs_to = to_times_back[1:] - hs_times_back[:-1]
        hs_hs = np.diff(hs_times_back)
        m = np.mean(hs_to / hs_hs)
        s = np.std(hs_to / hs_hs)

    res["BacRigMeanSwing"] = (m,
                              "Mean ratio of the swing phase relative to the "
                              "stride, for the right foot during the walk back")
    res["BacRigStdSwing"] = (s,
                             "Standard deviation of the ratio of the swing "
                             "phase relative to the stride, for the right foot "
                             "during the walk back")

    # Left foot
    hs_times = np.array(extract_left_hs_from_exo(e))
    hs_times_forward = hs_times[(hs_times > f_start) * (hs_times < f_end)]
    hs_times_back = hs_times[(hs_times > b_start) * (hs_times < b_end)]

    to_times = np.array(extract_left_to_from_exo(e))
    to_times_forward = to_times[(hs_times > f_start) * (hs_times < f_end)]
    to_times_back = to_times[(hs_times > b_start) * (hs_times < b_end)]

    # left foot forward
    if len(hs_times_forward) < 2:  # one or no step detected during this phase
        m = 0
        s = 0
        logging.warning("error for ForLefStdSwing and ForLefMeanSwing: "
                        + e.fname)
    else:
        hs_to = to_times_forward[1:] - hs_times_forward[:-1]
        hs_hs = np.diff(hs_times_forward)
        m = np.mean(hs_to / hs_hs)
        s = np.std(hs_to / hs_hs)

    res["ForLefMeanSwing"] = (m,
                              "Mean ratio of the swing phase relative to the "
                              "stride, for the left foot during the walk "
                              "forward")
    res["ForLefStdSwing"] = (s,
                             "Standard deviation of the ratio of the swing "
                             "phase relative to the stride, for the left foot "
                             "during the walk forward")
    # left foot back
    if len(hs_times_back) < 2:  # one or no step detected during this phase
        m = 0
        s = 0
        logging.warning("error for BacRigMeanSwing and BacRigStdSwing: "
                        + e.fname)
    else:
        hs_to = to_times_back[1:] - hs_times_back[:-1]
        hs_hs = np.diff(hs_times_back)
        m = np.mean(hs_to / hs_hs)
        s = np.std(hs_to / hs_hs)

    res["BacLefMeanSwing"] = (m,
                              "Mean ratio of the swing phase relative to the "
                              "stride, for the left foot during the walk back")
    res["BacLefStdSwing"] = (s,
                             "Standard deviation of the ratio of the swing "
                             "phase relative to the stride, for the left foot "
                             "during the walk back")

    return res


def feat_StepDuration(e):
    """
    Step duration defined as heel strike current foot --> heel strike opposite
    foot.
    """
    res = dict()

    # we extract the heel strike times
    hs_right = np.array(extract_right_hs_from_exo(e))
    hs_left = np.array(extract_left_hs_from_exo(e))
    # in case there is not the same number of steps in left and right
    L = min(len(hs_left), len(hs_right))
    hs_left = hs_left[:L]
    hs_right = hs_right[:L]

    ForRig = (
        e.seg_annotation[0], e.seg_annotation[1], "ForRig", "StepDur", hs_right,
        hs_left)
    ForLef = (
        e.seg_annotation[0], e.seg_annotation[1], "ForLef", "StepDur", hs_left,
        hs_right)
    BacRig = (
        e.seg_annotation[2], e.seg_annotation[3], "BacRig", "StepDur", hs_right,
        hs_left)
    BacLef = (
        e.seg_annotation[2], e.seg_annotation[3], "BacLef", "StepDur", hs_left,
        hs_right)

    for (start, end, phase, name, first_hs, second_hs) in (ForRig, ForLef,
                                                           BacRig, BacLef):
        tmp = (first_hs > start) * (first_hs < end)
        first_hs = first_hs[tmp] / fps
        second_hs = second_hs[tmp] / fps
        # 0 if it is the current foot, 1 if it is the opposite foot
        all_steps = list(zip(first_hs, [0] * len(first_hs))) + \
                    list(zip(second_hs, [1] * len(second_hs)))
        all_steps.sort(key=lambda x: x[0])

        step_durations = [t2 - t1 for (t1, d1), (t2, d2)
                          in zip(all_steps[:-1], all_steps[1:])
                          if d2 - d1 == 1]

        foot = "left"
        if "Rig" in phase:
            foot = "right"
        walk = "forward"
        if "Bac" in phase:
            walk = "back"

        res[phase + "Mean" + name] = (np.mean(step_durations),
                                      "Mean step duration (heel strike to heel"
                                      "strike of the opposite foot) for the " +
                                      foot + " foot during the walk " + walk +
                                      ".")
        res[phase + "Std" + name] = (np.std(step_durations),
                                     "Mean step duration (heel strike to heel"
                                     "strike of the opposite foot) for the " +
                                     foot + " foot during the walk " + walk +
                                     ".")
    return res


def feat_SingleSupport(e):
    """
    ratio of single support relative to stride
    (TO opposite foot --> HS opposite foot) / (HS current foot --> HS current foot)
    """
    res = dict()

    phase_dict = {"For": (e.seg_annotation[0], e.seg_annotation[1]),
                  "Bac": (e.seg_annotation[2], e.seg_annotation[3])}
    f = lambda x: np.array(x)
    foot_dict = {
        "Rig": (
            f(extract_right_to_from_exo(e)), f(extract_right_hs_from_exo(e)),
            f(extract_left_to_from_exo(e)), f(extract_left_hs_from_exo(e))),
        "Lef": (f(extract_left_to_from_exo(e)), f(extract_left_hs_from_exo(e)),
                f(extract_right_to_from_exo(e)),
                f(extract_right_hs_from_exo(e)))}

    for (phase, (start, end)), (
            foot,
            (current_to, current_hs, opposite_to, opposite_hs)) in product(
            phase_dict.items(), foot_dict.items()):
        # usually: HS_1 --> TO'_1 --> HS'_1 --> TO'_2 --> HS'_2 --> HS_2

        # hs current foot --> hs current foot
        tmp = (current_hs > start) * (current_hs < end)
        current_hs = np.array(current_hs[tmp]) / fps

        opposite_steps = list(zip(opposite_to / fps, cycle(["to"]))) + list(
            zip(
                opposite_hs / fps, cycle(["hs"])))
        opposite_steps.sort(key=lambda x: x[0])

        ratios = list()
        for hs1, hs2 in zip(current_hs[:-1], current_hs[1:]):
            ratios.append(sum(t2 - t1 for (t1, d1), (t2, d2) in
                              zip(opposite_steps[:-1], opposite_steps[1:])
                              if
                              hs1 < t1 < hs2 and hs1 < t2 < hs2 and d1 == "to"
                              and d2 == "hs") / (hs2 - hs1))

        foot_string, phase_string = "left", "forward"
        if foot == "Rig": foot_string = "right"
        if phase == "Bac": phase_string = "back"

        res[phase + foot + "Mean" + "SingSuppRatio"] = (
            np.mean(ratios),
            "Mean of the single support ratio relative to stride for the "
            + foot_string + " foot during the walk " + phase_string + ".")
        res[phase + foot + "Std" + "SingSuppRatio"] = (
            np.mean(ratios),
            "Std of the single support ratio relative to stride for the "
            + foot_string + " foot during the walk " + phase_string + ".")

    return res


def feat_DoubleSupport(e):
    """
    percentage of time spent in double support:
    sum(hs current foot --> to opposite foot) / (first hs --> last to)
    """
    res = dict()

    phase_dict = {"For": (e.seg_annotation[0], e.seg_annotation[1]),
                  "Bac": (e.seg_annotation[2], e.seg_annotation[3])}
    f = lambda x: np.array(x)
    foot_dict = {
        "Rig": (
            f(extract_right_to_from_exo(e)), f(extract_right_hs_from_exo(e)),
            f(extract_left_to_from_exo(e)), f(extract_left_hs_from_exo(e))),
        "Lef": (f(extract_left_to_from_exo(e)), f(extract_left_hs_from_exo(e)),
                f(extract_right_to_from_exo(e)),
                f(extract_right_hs_from_exo(e)))}

    for (phase, (start, end)), (
            foot,
            (current_to, current_hs, opposite_to, opposite_hs)) in product(
            phase_dict.items(), foot_dict.items()):
        tmp = (current_hs > start) * (current_hs < end)
        current_hs = np.array(current_hs[tmp]) / fps
        hs_to_succession = list(zip(current_hs, cycle(["hs"]))) + list(
            zip(opposite_to / fps, cycle(["to"])))
        hs_to_succession.sort(key=lambda x: x[0])

        double_support = sum(t2 - t1 for (t1, d1), (t2, d2) in
                             zip(hs_to_succession[:-1], hs_to_succession[1:])
                             if d1 == "hs" and d2 == "to") / (
                             max(t2 for (t1, d1), (t2, d2) in
                                 zip(hs_to_succession[:-1],
                                     hs_to_succession[1:]) if
                                 d1 == "hs" and d2 == "to") - min(
                                 t1 for (t1, d1), (t2, d2) in
                                 zip(hs_to_succession[:-1],
                                     hs_to_succession[1:]) if
                                 d1 == "hs" and d2 == "to"))

        foot_string, phase_string = "left", "forward"
        if foot == "Rig": foot_string = "right"
        if phase == "Bac": phase_string = "back"

        res[phase + foot + "DoubleSuppRatio"] = (
            double_support,
            "Ratio of the time spent in double support for the "
            + foot_string + " foot during the walk " + phase_string + ".")

    return res


def feat_StepCharacteristics(e):
    """
    step = hs current foot --> hs opposite foot
    mean, std, max, min, median on all signals, and mean and std of those values
    """
    res = dict()

    phase_dict = {"For": (e.seg_annotation[0], e.seg_annotation[1]),
                  "Bac": (e.seg_annotation[2], e.seg_annotation[3])}
    f = lambda x: np.array(x)
    foot_dict = {
        "Rig": (
            f(extract_right_to_from_exo(e)), f(extract_right_hs_from_exo(e)),
            f(extract_left_to_from_exo(e)), f(extract_left_hs_from_exo(e))),
        "Lef": (f(extract_left_to_from_exo(e)), f(extract_left_hs_from_exo(e)),
                f(extract_right_to_from_exo(e)),
                f(extract_right_hs_from_exo(e)))}

    # characteristics
    charact = {"Max": np.max, "Min": np.min, "Mean": np.mean, "Std": np.std,
               "Med": np.median}
    charact_doc = {"Max": "maximums", "Min": "minimums", "Mean": "means",
                   "Std": "standard deviations", "Med": "medians"}
    # we get all the signals
    axes = {ax + sensor: e.data_sensor[k * 6 + i] for
            (k, sensor), (i, ax) in
            product(enumerate(['RightFoot', 'LeftFoot', 'Waist', 'Head']),
                    enumerate(["AccX", "AccY", "AccZ", "GyrX",
                               "GyrY", "GyrZ"]))}
    # we add the calibrated signals
    for (k, sensor) in enumerate(['RightFoot', 'LeftFoot', 'Waist', 'Head']):
        axes["AccV" + sensor] = e.data_earth[6 * k + 2]
        axes["GyrV" + sensor] = e.data_earth[6 * k + 5]

    for (phase, (start, end)), (
            foot,
            (current_to, current_hs, opposite_to, opposite_hs)) in product(
            phase_dict.items(), foot_dict.items()):
        # for the documentation
        foot_string, phase_string = "left", "forward"
        if foot == "Rig": foot_string = "right"
        if phase == "Bac": phase_string = "back"

        # we keep the steps which are in the considered phase
        tmp = (current_hs > start) * (current_hs < end)
        all_steps = list(zip(current_hs[tmp], cycle(["current"]))) + list(
            zip(opposite_hs, cycle(["opposite"])))
        all_steps.sort(key=lambda x: x[0])

        step_times = [(t1, t2) for (t1, d1), (t2, d2) in
                      zip(all_steps[:-1], all_steps[1:]) if
                      d1 == "current" and d2 == "opposite"]
        # we iterate for all dimensions and sensors
        for (dimension, signal), (func_name, func) in product(axes.items(),
                charact.items()):
            values = [func(signal[t1:t2]) for t1, t2 in step_times if t1 != t2]
            m, s = np.mean(values), np.std(values)

            res[phase + "Mean" + dimension + "Step" + func_name] = \
                (m,
                 "Mean of the signal "
                 "{} {} during each step of the {} foot of the walk {}.".format(
                     dimension, charact_doc[func_name], foot_string,
                     phase_string))
            res[phase + "Std" + dimension + "Step" + func_name] = \
                (s,
                 "Standard deviation of the signal "
                 "{} {} during each step of the {} foot of the walk {}.".format(
                     dimension, charact_doc[func_name], foot_string,
                     phase_string))
    return res


def feat_StrideCharacteristics(e):
    """
    stride = hs current foot --> hs current foot
    mean, std, max, min, median on all signals, and mean and std of those values
    """
    res = dict()

    phase_dict = {"For": (e.seg_annotation[0], e.seg_annotation[1]),
                  "Bac": (e.seg_annotation[2], e.seg_annotation[3])}
    f = lambda x: np.array(x)
    foot_dict = {
        "Rig": (
            f(extract_right_to_from_exo(e)), f(extract_right_hs_from_exo(e)),
            f(extract_left_to_from_exo(e)), f(extract_left_hs_from_exo(e))),
        "Lef": (f(extract_left_to_from_exo(e)), f(extract_left_hs_from_exo(e)),
                f(extract_right_to_from_exo(e)),
                f(extract_right_hs_from_exo(e)))}

    # characteristics
    charact = {"Max": np.max, "Min": np.min, "Mean": np.mean, "Std": np.std,
               "Med": np.median}
    charact_doc = {"Max": "maximums", "Min": "minimums", "Mean": "means",
                   "Std": "standard deviations", "Med": "medians"}
    # we get all the signals
    axes = {ax + sensor: e.data_sensor[k * 6 + i] for
            (k, sensor), (i, ax) in
            product(enumerate(['RightFoot', 'LeftFoot', 'Waist', 'Head']),
                    enumerate(["AccX", "AccY", "AccZ", "GyrX",
                               "GyrY", "GyrZ"]))}
    # we add the calibrated signals
    for (k, sensor) in enumerate(['RightFoot', 'LeftFoot', 'Waist', 'Head']):
        axes["AccV" + sensor] = e.data_earth[6 * k + 2]
        axes["GyrV" + sensor] = e.data_earth[6 * k + 5]

    for (phase, (start, end)), (
            foot,
            (current_to, current_hs, opposite_to, opposite_hs)) in product(
            phase_dict.items(), foot_dict.items()):
        # for the documentation
        foot_string, phase_string = "left", "forward"
        if foot == "Rig": foot_string = "right"
        if phase == "Bac": phase_string = "back"

        # we keep the steps which are in the considered phase
        tmp = (current_hs > start) * (current_hs < end)
        all_steps = current_hs[tmp]
        step_times = [(t1, t2) for t1, t2 in
                      zip(all_steps[:-1], all_steps[1:])]
        # we iterate for all dimensions and sensors
        for (dimension, signal), (func_name, func) in product(axes.items(),
                charact.items()):
            values = [func(signal[t1:t2]) for t1, t2 in step_times if t1 != t2]
            m, s = np.mean(values), np.std(values)

            res[phase + "Mean" + dimension + "Stride" + func_name] = \
                (m,
                 "Mean of the signal "
                 "{} {} during each stride of the {} foot of the walk {}.".format(
                     dimension, charact_doc[func_name], foot_string,
                     phase_string))
            res[phase + "Std" + dimension + "Stride" + func_name] = \
                (s,
                 "Standard deviation of the signal "
                 "{} {} during each stride of the {} foot of the walk {}.".format(
                     dimension, charact_doc[func_name], foot_string,
                     phase_string))
    return res

#
# if __name__ == '__main__':
#     from db_marche import Database
#
#     db = Database()  # on charge la base
#     # on en extrait 10 exercices (seed pour la reproductibilité)
#     exos = db.get_data(limit=10, seed=123)
#
#     # for k, ex in enumerate(exos):
#     #     print(k)
#     #     feat_StepCharacteristics(ex)
#     #     feat_StrideCharacteristics(ex)
#     #     feat_StrideDuration(ex)
#     #     feat_DoubleSupport(ex)
#     #     feat_SingleSupport(ex)
#     #     feat_StepNum(ex)
#     #     feat_SwingPhase(ex)
#     ex = exos[8]
#     for keys, values in feat_StepNum(ex).items():
#         # if "Mean" in keys:
#         # if "Std" in keys:
#         print(keys)
#         print(values)
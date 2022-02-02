# Text will be segmented into 20 key sections accordingto the section titles.
# The sections will be combined into 10 key chunks as below
# 1. sex, ["service:", "allergies:", "chief complaint:", "major surgical or invasive procedure:"],
# 2. ["history of present illness:"],
# 3. ["past medical history:", "past surgical history:", "social history:", "family history:"],
# 4. ["physical exam:", "physical examination:"],
# 5. ["pertinent results:", "laboratory data:", "initial laboratory:"],
# 6. ["brief hospital course:", "hospital course:"],
# 7. ["medications on admission:", "discharge medications:", "medications on discharge:"],
# 8. ["discharge disposition:", "discharge diagnosis:", "discharge diagnoses:", "discharge condition:",
# #                    "condition on discharge:", "condition at discharge:"],
# 9. ["discharge instructions:", "followup instructions:"],
# 10. ['addendum:']
import csv
import re
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, RegexpTokenizer

# keep only alphanumeric
tokenizer = RegexpTokenizer(r'\w+')

stop_words = set(stopwords.words('english'))

section_titles = {"service:": "service",
                  "allergies:": "allergies",
                  "chief complaint:": "chief_complaint",
                  "major surgical or invasive procedure:": "major_procedure",
                  "history of present illness:": "history_present_illness",
                  "past medical history:": "past_medical_history",
                  "past surgical history:": "past_surgical_history",
                  "social history:": "social_history",
                  "family history:": "family_history",
                  "physical exam:": "physical_exam",
                  "physical examination:": "physical_exam",
                  "pertinent results:": "pertinent_results",
                  "laboratory data:": "pertinent_results",
                  "initial laboratory:": "pertinent_results",
                  "brief hospital course:": "hospital_course",
                  "hospital course:": "hospital_course",
                  "summary of hospital course:": "hospital_course",
                  "medications on admission:": "medications_on_admission",
                  "discharge medications:": "discharge_medications",
                  "medications on discharge:": "discharge_medications",
                  "discharge disposition:": "discharge_disposition",
                  "discharge diagnosis:": "discharge_diagnosis",
                  "discharge diagnoses:": "discharge_diagnosis",
                  "preoperative diagnosis:": "discharge_diagnosis",
                  "postoperative diagnosis:": "discharge_diagnosis",
                  "diagnosis:": "discharge_diagnosis",
                  "surgical procedure:": "procedure",
                  "procedure:": "procedure",
                  "discharge condition:": "discharge_condition",
                  "condition on discharge:": "discharge_condition",
                  "condition at discharge:": "discharge_condition",
                  "discharge instructions:": "discharge_instructions",
                  "followup instructions:": "followup_instructions",
                  "addendum:": "addendum"
                  }


class DischargeSummary:
    def __init__(self):
        self.sex = None
        self.service = None
        self.allergies = None
        self.chief_complaint = None
        self.major_procedure = None
        self.history_present_illness = None
        self.past_medical_history = None
        self.past_surgical_history = None
        self.social_history = None
        self.family_history = None
        self.physical_exam = None
        self.pertinent_results = None
        self.hospital_course = None
        self.medications_on_admission = None
        self.discharge_medications = None
        self.discharge_disposition = None
        self.discharge_diagnosis = None
        self.discharge_condition = None
        self.discharge_instructions = None
        self.followup_instructions = None

    def set_section_text(self, section_title, text):
        if getattr(self, section_title) is not None:
            text = "\n".join([getattr(self, section_title), text])

        setattr(self, section_title, text)


class MainDischargeSummary(DischargeSummary):
    def __init__(self):
        super().__init__()
        # combine the same sections from addendum into main discharge summaries
        self.addendum_object = []
        self.addendum = None

    def get_section_text(self, section_title):
        # merge the same sections from all the discharge summaries
        main_section_text = getattr(self, section_title)
        if main_section_text is not None:
            main_section_text = main_section_text.split("\n")
        else:
            main_section_text = []
        for add_ds in self.addendum_object:
            add_ds_section_text = getattr(add_ds, section_title)
            if add_ds_section_text is not None:
                for line in add_ds_section_text.split("\n"):
                    if line not in main_section_text:
                        main_section_text.append(line)

        # remove the duplicated
        main_section_text = [line for line in main_section_text if len(line) != 0]

        return "\n".join(main_section_text)


class AddendumDischargeSummary(DischargeSummary):
    def __init__(self):
        super().__init__()
        self.addendum = None


def parse_discharge_summaries(text):
    # check if there are addendum discharge summaries. The first discharge summaries will be
    # the main discharge summaries and the rest will be the addendum.
    # input: discharge summaries for one admission
    discharge_summaries = re.split(r"\[EOD2022\]", text)
    # extract sections from main and addendum discharge summaries
    ds_object = MainDischargeSummary()
    main_ds_index = None
    for index, ds in enumerate(discharge_summaries):
        if len(ds) == 0:
            continue
        else:
            if main_ds_index is None:
                main_ds_index = index
                ds_type = "main"
            else:
                ds_type = "addendum"
                ds_addendum_object = AddendumDischargeSummary()

        # scan lines of text to extract the sections one by one
        current_section = None
        section_start_line = None
        ds_content = ds.split("\n")
        for i, line in enumerate(ds_content):
            # save Sex information
            match = re.search("date of birth:\s*sex:", line)
            if match is not None and ds_type == "main":
                setattr(ds_object, "sex", line)
                continue

            match = re.search(r"\w+:", line)
            if match is None:
                continue
            possible_section_title = line[:match.end()]
            if possible_section_title == "addendum:":
                if current_section in ["pertinent results:", "laboratory data:",
                                       "initial laboratory:", "followup instructions:",
                                       "brief hospital course:", "diagnosis:",
                                       "summary of hospital course:", "discharge medications:",
                                       "medications on discharge:"]:
                    # addendum under pertinent results
                    continue
                else:
                    if current_section not in ["service:", "allergies:"]:
                        print("find addendum unusual")


            if possible_section_title in section_titles.keys():
                if current_section is not None:
                    # save current section to object
                    section_content = "\n".join(ds_content[section_start_line: i])
                    if ds_type == "main":
                        setattr(ds_object, section_titles.get(current_section), section_content)
                    else:
                        setattr(ds_addendum_object, section_titles.get(current_section), section_content)

                # reset to current section
                current_section = possible_section_title
                section_start_line = i
        # save the last section of discharge summary
        if current_section is not None:
            section_content = "\n".join(ds_content[section_start_line:])
            if ds_type == "main":
                setattr(ds_object, section_titles.get(current_section), section_content)
            else:
                setattr(ds_addendum_object, section_titles.get(current_section), section_content)

        if ds_type == "addendum":
            ds_object.addendum_object.append(ds_addendum_object)

    return pd.Series([ds_object.get_section_text("sex"),
                      ds_object.get_section_text("service"),
                      ds_object.get_section_text("allergies"),
                      ds_object.get_section_text("chief_complaint"),
                      ds_object.get_section_text("major_procedure"),
                      ds_object.get_section_text("history_present_illness"),
                      ds_object.get_section_text("past_medical_history"),
                      ds_object.get_section_text("past_surgical_history"),
                      ds_object.get_section_text("social_history"),
                      ds_object.get_section_text("family_history"),
                      ds_object.get_section_text("physical_exam"),
                      ds_object.get_section_text("pertinent_results"),
                      ds_object.get_section_text("hospital_course"),
                      ds_object.get_section_text("medications_on_admission"),
                      ds_object.get_section_text("discharge_medications"),
                      ds_object.get_section_text("discharge_disposition"),
                      ds_object.get_section_text("discharge_diagnosis"),
                      ds_object.get_section_text("discharge_condition"),
                      ds_object.get_section_text("discharge_instructions"),
                      ds_object.get_section_text("followup_instructions"),
                      ds_object.get_section_text("addendum"),
                      ])

def contains_alphabetic(token):
    for c in token:
        if c.isalpha():
            return True
    return False

def normalise_text(text):
    output = []
    length = 0

    for sent in sent_tokenize(text):
        tokens = [token.lower() for token in tokenizer.tokenize(sent) if contains_alphabetic(token)]
        length += len(tokens)

        sent = " ".join(tokens)

        if len(sent) > 0:
            output.append(sent)

    return "\n".join(output), length

def preprocess_level_1(text):
    # remove non-alphabetic characters
    return normalise_text(text)[0]


def preprocess_level_2(text):
    text = preprocess_level_1(text)
    tokens = [w for w in text.split() if not w.lower() in stop_words]
    return " ".join(tokens)


def segment_ds(input_dir, FLAGS, logging, data_type="full", task='mimic3'):
    logging.info("Datasets from %s", input_dir)

    if task == "mimic3":
        train_dataset = pd.read_csv(Path(input_dir) / "train_{}.csv".format(data_type))
        dev_dataset = pd.read_csv(Path(input_dir) / "dev_{}.csv".format(data_type))
        test_dataset = pd.read_csv(Path(input_dir) / "test_{}.csv".format(data_type))
    else:
        # mimic2
        train_dataset = pd.read_csv(Path(input_dir) / "train_{}.csv".format(data_type), encoding='ISO-8859-1')
        dev_dataset = pd.read_csv(Path(input_dir) / "dev_{}.csv".format(data_type), encoding='ISO-8859-1')
        test_dataset = pd.read_csv(Path(input_dir) / "test_{}.csv".format(data_type), encoding='ISO-8859-1')


    train_size = train_dataset.shape[0]
    dev_size = dev_dataset.shape[0]
    test_size = test_dataset.shape[0]

    all_data = train_dataset.append(dev_dataset, ignore_index=True).append(test_dataset, ignore_index=True)

    logging.info("All data size %s", all_data.shape)
    logging.info("Train size %s", train_size)
    logging.info("Dev size %s", dev_size)
    logging.info("Test size %s", test_size)

    # 1) remove \n character and join the words with whitespaces
    def remove_newline_character(text):
        return " ".join([item for item in text.split("\n") if len(item) != 0])

    if FLAGS.segment_text:
        logging.info("Divide the text to 21 sections")
        logging.info("Segment text into pre-defined meaningful chunks")
        all_data[["sex",
                  "service",
                  "allergies",
                  "chief_complaint",
                  "major_procedure",
                  "history_present_illness",
                  "past_medical_history",
                  "past_surgical_history",
                  "social_history",
                  "family_history",
                  "physical_exam",
                  "pertinent_results",
                  "hospital_course",
                  "medications_on_admission",
                  "discharge_medications",
                  "discharge_disposition",
                  "discharge_diagnosis",
                  "discharge_condition",
                  "discharge_instructions",
                  "followup_instructions",
                  "addendum"]] = all_data["Text"].apply(parse_discharge_summaries)

        if FLAGS.pre_process_level == "level_1":
            # preprocess_level_1: remove non-alphabetic words
            all_data["service"] = all_data["service"].apply(preprocess_level_1)
            all_data["allergies"] = all_data["allergies"].apply(preprocess_level_1)
            all_data["chief_complaint"] = all_data["chief_complaint"].apply(preprocess_level_1)
            all_data["major_procedure"] = all_data["major_procedure"].apply(preprocess_level_1)
            all_data["history_present_illness"] = all_data["history_present_illness"].apply(preprocess_level_1)
            all_data["past_medical_history"] = all_data["past_medical_history"].apply(preprocess_level_1)
            all_data["past_surgical_history"] = all_data["past_surgical_history"].apply(preprocess_level_1)
            all_data["social_history"] = all_data["social_history"].apply(preprocess_level_1)
            all_data["family_history"] = all_data["family_history"].apply(preprocess_level_1)
            all_data["physical_exam"] = all_data["physical_exam"].apply(preprocess_level_1)
            all_data["pertinent_results"] = all_data["pertinent_results"].apply(preprocess_level_1)
            all_data["hospital_course"] = all_data["hospital_course"].apply(preprocess_level_1)
            all_data["medications_on_admission"] = all_data["medications_on_admission"].apply(preprocess_level_1)
            all_data["discharge_medications"] = all_data["discharge_medications"].apply(preprocess_level_1)
            all_data["discharge_disposition"] = all_data["discharge_disposition"].apply(preprocess_level_1)
            all_data["discharge_diagnosis"] = all_data["discharge_diagnosis"].apply(preprocess_level_1)
            all_data["discharge_condition"] = all_data["discharge_condition"].apply(preprocess_level_1)
            all_data["discharge_instructions"] = all_data["discharge_instructions"].apply(preprocess_level_1)
            all_data["followup_instructions"] = all_data["followup_instructions"].apply(preprocess_level_1)
            all_data["addendum"] = all_data["addendum"].apply(preprocess_level_1)
            logging.info("Level 1 pre-processing has been done.")

        if FLAGS.pre_process_level == "level_2":
            # preprocess_level_2: remove stop words
            all_data["service"] = all_data["service"].apply(preprocess_level_2)
            all_data["allergies"] = all_data["allergies"].apply(preprocess_level_2)
            all_data["chief_complaint"] = all_data["chief_complaint"].apply(preprocess_level_2)
            all_data["major_procedure"] = all_data["major_procedure"].apply(preprocess_level_2)
            all_data["history_present_illness"] = all_data["history_present_illness"].apply(preprocess_level_2)
            all_data["past_medical_history"] = all_data["past_medical_history"].apply(preprocess_level_2)
            all_data["past_surgical_history"] = all_data["past_surgical_history"].apply(preprocess_level_2)
            all_data["social_history"] = all_data["social_history"].apply(preprocess_level_2)
            all_data["family_history"] = all_data["family_history"].apply(preprocess_level_2)
            all_data["physical_exam"] = all_data["physical_exam"].apply(preprocess_level_2)
            all_data["pertinent_results"] = all_data["pertinent_results"].apply(preprocess_level_2)
            all_data["hospital_course"] = all_data["hospital_course"].apply(preprocess_level_2)
            all_data["medications_on_admission"] = all_data["medications_on_admission"].apply(preprocess_level_2)
            all_data["discharge_medications"] = all_data["discharge_medications"].apply(preprocess_level_2)
            all_data["discharge_disposition"] = all_data["discharge_disposition"].apply(preprocess_level_2)
            all_data["discharge_diagnosis"] = all_data["discharge_diagnosis"].apply(preprocess_level_2)
            all_data["discharge_condition"] = all_data["discharge_condition"].apply(preprocess_level_2)
            all_data["discharge_instructions"] = all_data["discharge_instructions"].apply(preprocess_level_2)
            all_data["followup_instructions"] = all_data["followup_instructions"].apply(preprocess_level_2)
            all_data["addendum"] = all_data["addendum"].apply(preprocess_level_2)
            logging.info("Level 2 pre-processing has been done.")

        logging.info("Combine 21 sections to 10 pre-defined meaningful chunks.")

        all_data["Chunk1"] = (all_data["sex"].map(str) + "\n"
                              + all_data["service"].map(str) + "\n"
                              + all_data["allergies"].map(str) + "\n"
                              + all_data["chief_complaint"].map(str) + "\n"
                              + all_data["major_procedure"].map(str)).apply(remove_newline_character)
        all_data["Chunk2"] = (all_data["history_present_illness"]).apply(remove_newline_character)
        all_data["Chunk3"] = (all_data["past_medical_history"].map(str) + "\n"
                              + all_data["past_surgical_history"].map(str) + "\n"
                              + all_data["social_history"].map(str) + "\n"
                              + all_data["family_history"].map(str)).apply(remove_newline_character)
        all_data["Chunk4"] = (all_data["physical_exam"]).apply(remove_newline_character)
        all_data["Chunk5"] = (all_data["pertinent_results"]).apply(remove_newline_character)
        all_data["Chunk6"] = (all_data["hospital_course"]).apply(remove_newline_character)
        all_data["Chunk7"] = (all_data["medications_on_admission"].map(str) + "\n"
                              + all_data["discharge_medications"].map(str)).apply(remove_newline_character)
        all_data["Chunk8"] = (all_data["discharge_disposition"].map(str) + "\n"
                              + all_data["discharge_diagnosis"].map(str) + "\n"
                              + all_data["discharge_condition"].map(str)).apply(remove_newline_character)
        all_data["Chunk9"] = (all_data["discharge_instructions"].map(str) + "\n"
                              + all_data["followup_instructions"].map(str)).apply(remove_newline_character)
        all_data["Chunk10"] = (all_data["addendum"]).apply(remove_newline_character)
    else:
        logging.info("Not segment text")
        def process_text(text):
            # splits text to multiple discharge summaries
            discharge_summaries = re.split(r"\[EOD2022\]", text)
            return "\n".join(discharge_summaries)

        all_data["processed_text"] = all_data["Text"].apply(process_text)
        if FLAGS.pre_process_level == "level_1":
            all_data["processed_text"] = all_data["processed_text"].apply(preprocess_level_1)
        if FLAGS.pre_process_level == "level_2":
            all_data["processed_text"] = all_data["processed_text"].apply(preprocess_level_2)
        all_data["processed_text"] = all_data["processed_text"].apply(remove_newline_character)

    # add multi-hot labels columns
    def transform_labels(labels):
        labels = labels.split('|')
        return tuple(labels)

    labels = all_data['Full_Labels']
    mlb = MultiLabelBinarizer()
    mlb.fit([[label for item in labels.apply(transform_labels).tolist() for label in item]])

    multi_hot_labels = mlb.transform(labels.apply(transform_labels))

    new_data = pd.DataFrame()
    new_data["hadm_id"] = all_data["Admission_Id"]
    logging.info("Convert Full_Labels to multi-hot labels")
    if FLAGS.segment_text:
        labels = pd.DataFrame(multi_hot_labels, columns=mlb.classes_)
        new_data = pd.concat([new_data, all_data.loc[:, all_data.columns.str.startswith("Chunk")], labels], axis=1)
    else:
        all_data["multi_hot_labels"] = pd.DataFrame(multi_hot_labels, columns=mlb.classes_).apply(
            lambda x: [label for label in x], axis=1)
        new_data["text"] = all_data["processed_text"]
        new_data["labels"] = all_data["multi_hot_labels"]

    # split data into 3 sets
    train_data = new_data.iloc[:train_size, :].reset_index(drop=True)
    dev_data = new_data.iloc[train_size:(train_size + dev_size), :].reset_index(drop=True)
    test_data = new_data.iloc[(train_size + dev_size):, :].reset_index(drop=True)

    train_file = Path(input_dir) / "train_data_{}_{}.csv".format(data_type, FLAGS.pre_process_level)
    dev_file = Path(input_dir) / "dev_data_{}_{}.csv".format(data_type, FLAGS.pre_process_level)
    test_file = Path(input_dir) / "test_data_{}_{}.csv".format(data_type, FLAGS.pre_process_level)

    train_data.to_csv(train_file, index=False)
    logging.info("Train data saved to %s", train_file)
    dev_data.to_csv(dev_file, index=False)
    logging.info("Dev data saved to %s", dev_file)
    test_data.to_csv(test_file, index=False)
    logging.info("Test data saved to %s", test_file)

    if task == 'mimic3':
        # Get label dictionary
        labels_dictionary = {}
        for index, value in all_data["Labels_Dictionary"].items():
            labels_dictionary.update(eval(value))

    # write label dicationary to csv
    label_file = Path(input_dir) / "labels_dictionary_{}_{}.csv".format(data_type, FLAGS.pre_process_level)
    with open(label_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["icd9_code", "long_title"])
        for code in mlb.classes_:
            if task == 'mimic3':
                writer.writerow([code, labels_dictionary[code]])
            else:
                writer.writerow([code, ""])

    logging.info("Label Dictionary saved to %s", label_file)






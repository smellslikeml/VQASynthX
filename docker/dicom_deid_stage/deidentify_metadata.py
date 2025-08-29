# Jupyter notebook converted to Python script.
import os
import pydicom
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.data import get_testdata_file
import re
import hashlib
import csv
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from shutil import copyfile
from pydicom.valuerep import IS

# 需要remove的tags(不包含需要删除部分内容的tag)
tags_to_clear = [
    (0x0008, 0x0050),  # Accession Number
    (0x0008, 0x0054),  # Retrieve AE Title
    (0x0008, 0x0080),  # Institution Name
    (0x0008, 0x0081),  # Institution Address
    (0x0008, 0x0090),  # Referring Physician Name
    (0x0008, 0x0201),  # Timezone Offset From UTC
    (0x0008, 0x1010),  # Station Name
    (0x0008, 0x1048),  # Physician(s) of Record
    (0x0008, 0x1050),  # Performing Physician's Name
    (0x0008, 0x1070),  # Operators' Name
    (0x0008, 0x1120),  # Referenced Patient Sequence
    (0x0010, 0x0030),  # Patient Birth Date
    (0x0010, 0x1000),
    (0x0010, 0x1040),
    (0x0012, 0x0020),
    (0x0012, 0x0021),
    (0x0012, 0x0030),
    (0x0012, 0x0031),
    (0x0012, 0x0040),
    (0x0012, 0x0042),
    (0x0012, 0x0050),
    (0x0018, 0x1200),
    (0x0020, 0x0010),
    (0x0032, 0x1021),
    (0x0040, 0x0009),
    (0x0040, 0x0241),
    (0x0040, 0x0275),
    (0x0070, 0x0084),
    (0x0088, 0x0220),
    (0x0040, 0xA075),
    (0x0040, 0xA123),
    (0x0012, 0x0051),
    (0x0043, 0x1005),
    (0x0043, 0x1029),
    (0x0043, 0x1060),
    (0x0043, 0x1080),
    (0x0009, 0x1002),
    (0x0009, 0x1030),
    (0x0009, 0x1037),
    (0x0018, 0xA001),
    (0x0008, 0x1040),
    (0x0029, 0x1131),
    (0x0029, 0x1134),
    (0x0040, 0xA027),
    (0x0040, 0xA073),
    (0x0040, 0xA730),
    (0x0021, 0x1035),
    (0x0021, 0x1003),
    (0x0040, 0x1001),
    (0x0032, 0x1020),
    (0x0040, 0x0242),
    (0x0040, 0x0280),
    (0x0040, 0x2001),
]
tags_to_shift = [
    0x00080012,
    0x00080020,
    0x00080021,
    0x00080022,
    0x00080023,
    0x00080024,
    0x00080025,
    0x0008002A,
    0x001021D0,
    0x00181012,
    0x0018700C,
    0x00380020,
    0x00400002,
    0x00400004,
    0x00400244,
    0x00400250,
    0x0040A032,
    0x00091005,
    0x0009100D,
    0x0009100E,
    0x00540016,
    0x00540410,
    0x00540414,
    0x0040A730,
    0x0040A073,
    0x0018A001,
    0x0018A002,
    0x0019109D,
    0x00091039,
    0x0009103B,
    0x0009103D,
    0x00091068,
    0x0009106C,
    0x0009107B,
    0x000910E9,
    0x00171004,  # discrepancy,
    0x00181078,
    0x00181079,
    0x0040A030,
    0x0040A120,
    0x0040A121,
    0x00540300,
    0x00540412,
    0x00080106,
    0x00191010,
    0x00321040,
    0x00321050,
]
tags_to_hash = [
    0x0020000D,
    0x0020000E,
    0x00200052,
    0x00200200,
    0x0040A124,
    0x00880140,
    0x00400555,
    0x0040A073,
    0x0040A730,
    0x00080014,
    0x00080018,
    0x00080118,
    0x00081155,
    0x00081120,
    0x00083010,
    0x00081110,
    0x00081111,
    0x00081140,
    0x00082112,
    0x00081250,
    0x00089121,
    0x00400513,
    0x00400562,
    0x00400610,
    0x00340001,
    0x00081084,
    0x0009100A,
    0x00091013,
    0x00091056,
    0x00091057,
    0x00091059,
    0x0009105C,
    0x0009105D,
    0x0009105E,
    0x00091098,
    0x00091097,
    0x000910AD,
    0x00091007,
    0x000910E3,
    0x00431088,
    0x00431098,
    0x00451050,
    0x00451051,
    0x0008010C,  # discrepancy
    0x00080110,
    0x0040A375,
    0x00081199,
    0x0040A504,
    0x00081115,
]  # discrepancy sequence
tag_to_add = [
    0x00080068,
    0x00180060,
    0x00185101,
    0x00187004,
    0x00200012,
    0x00201040,
    0x20500020,
]
tag_not_null = [0x00080068, 0x00181000, 0x00200062, 0x20500020, 0x00402001]
sensitive_tags = [
    0x00081030,
    0x0008103E,
    0x001021B0,
    0x00102000,
    0x00104000,
    0x00181030,
    0x00184000,
    0x00204000,
    0x00321060,
    0x00324000,
    0x011710C4,
    0x01171024,
    0x00400275,
    0x00400007,
    0x00401400,
    0x00400310,
    0x00200011,
    0x0040A730,
    0x0040A160,
]


def set_empty_tags_to_add(ds, tags):
    for tag in tags:
        tag = pydicom.tag.Tag(tag)
        if tag in ds:
            value = ds[tag].value
            if (
                value is None
                or (isinstance(value, str) and value.strip() == "")
                or (isinstance(value, list) and len(value) == 0)
            ):
                ds[tag].value = "NA"


def apply_id_mapping(ds, id_lookup):
    if "PatientID" in ds:
        original_id = ds.PatientID
        new_id = id_lookup.get(original_id, original_id)
        ds.PatientID = new_id


def modify_patient_name_to_id(ds):
    if "PatientID" in ds and "PatientName" in ds:
        patient_id = ds.PatientID
        ds.PatientName = patient_id


def clear_tags(ds, tags_to_clear):
    if isinstance(ds, Dataset):
        for elem in ds:
            if isinstance(elem.value, Sequence):
                for item in elem.value:
                    if isinstance(item, Dataset):
                        clear_tags(item, tags_to_clear)
            elif isinstance(elem.value, Dataset):
                clear_tags(elem.value, tags_to_clear)
            elif elem.tag in tags_to_clear:
                try:
                    value = ds[elem.tag].value
                    if isinstance(value, str):
                        ds[elem.tag].value = ""
                    elif isinstance(value, (int, float, bool)):
                        ds[elem.tag].value = None
                    else:
                        ds[elem.tag].value = b"" if isinstance(value, bytes) else None
                except Exception as e:
                    print(f"Error processing tag {elem.tag}: {e}")


def load_offsets_from_csv(shift_csv_path):
    offsets = {}
    try:
        with open(shift_csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) != 2:
                    continue
                patient_id, offset_str = row
                try:
                    offset = int(offset_str)
                    offsets[patient_id] = offset
                except ValueError:
                    print(
                        f"Warning: Invalid offset value '{offset_str}' for PatientID '{patient_id}'. Skipping."
                    )
    except FileNotFoundError:
        print(
            f"Warning: Shift CSV file not found at {shift_csv_path}. No date shifting will be performed."
        )
    return offsets


def add_missing_tags(ds, tags, vr="LO"):
    for tag in tags:
        tag_tuple = (tag >> 16, tag & 0xFFFF)
        if tag_tuple not in ds:
            ds.add_new(tag_tuple, vr, "")


def shift_datetime(date_str, days):
    try:
        if not isinstance(days, int):
            raise ValueError(f"Date increment should be an integer, got {type(days)}")
        if isinstance(date_str, str):
            if len(date_str) == 8:
                dt = datetime.strptime(date_str, "%Y%m%d")
                shifted_dt = dt - timedelta(days=days)
                return shifted_dt.strftime("%Y%m%d")
            elif len(date_str) == 14:
                dt = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                shifted_dt = dt - timedelta(days=days)
                return shifted_dt.strftime("%Y%m%d%H%M%S")
            elif len(date_str) == 17:
                date_part = date_str[:14]
                fractional_part = date_str[14:]
                dt = datetime.strptime(date_part, "%Y%m%d%H%M%S")
                shifted_dt = dt - timedelta(days=days)
                return shifted_dt.strftime("%Y%m%d%H%M%S") + fractional_part
            elif len(date_str) == 10 and date_str.isdigit():
                timestamp = int(date_str)
                dt = datetime.utcfromtimestamp(timestamp)
                shifted_dt = dt - timedelta(days=days)
                return str(int(shifted_dt.timestamp()))
            else:
                return date_str
        elif isinstance(date_str, int):
            dt = datetime.utcfromtimestamp(date_str)
            shifted_dt = dt + timedelta(days=days)
            return str(int(shifted_dt.timestamp()))
        else:
            return date_str
    except Exception:
        return date_str


def update_tag(ds, tag, date_increment):
    if tag in ds:
        old_value = ds[tag].value
        if not isinstance(date_increment, int):
            return
        if isinstance(old_value, int):
            old_value = str(old_value)
        elif not isinstance(old_value, str):
            return
        new_value = shift_datetime(old_value, date_increment)
        if isinstance(ds[tag].value, str):
            ds[tag].value = new_value
        elif isinstance(ds[tag].value, int) and new_value.isdigit():
            ds[tag].value = int(new_value)


def process_item(item, tags_to_shift, date_increment):
    if isinstance(item, Dataset):
        for tag in tags_to_shift:
            update_tag(item, tag, date_increment)
        for element in item:
            if isinstance(element.value, (Dataset, Sequence)):
                process_item(element.value, tags_to_shift, date_increment)
    elif isinstance(item, Sequence):
        for sub_item in item:
            process_item(sub_item, tags_to_shift, date_increment)


def shift_dates(ds, tags_to_shift, patient_offsets):
    try:
        if (0x0010, 0x0020) in ds:
            patient_id = ds[(0x0010, 0x0020)].value
            date_increment = patient_offsets.get(patient_id, 0)
            for tag in tags_to_shift:
                if tag in ds:
                    if isinstance(ds[tag].value, (Dataset, Sequence)):
                        process_item(ds[tag].value, tags_to_shift, date_increment)
                    else:
                        update_tag(ds, tag, date_increment)
    except Exception as e:
        print(f"Error shifting dates: {e}")


def hash_uid(uid):
    hashed_uid = hashlib.sha256(uid.encode()).hexdigest()
    numeric_hash = int(hashed_uid, 16)
    numeric_hash_str = str(numeric_hash).zfill(19)[:19]
    return numeric_hash_str


def hash_sequence(ds, tags_to_hash, uid_root, uid_mapping, patient_id=None):
    if isinstance(ds, pydicom.Dataset):
        for tag in tags_to_hash:
            if tag in ds and ds[tag].value:
                if isinstance(ds[tag].value, pydicom.sequence.Sequence):
                    for item in ds[tag].value:
                        hash_sequence(
                            item, tags_to_hash, uid_root, uid_mapping, patient_id
                        )
                else:
                    original_uid = ds[tag].value
                    uid_root_with_patient = f"{uid_root}{patient_id}.8.117."
                    if original_uid in uid_mapping:
                        new_uid = uid_mapping[original_uid]
                        ds[tag].value = new_uid
                    else:
                        new_uid = hash_uid(original_uid)
                        full_new_uid = f"{uid_root_with_patient}{new_uid}"
                        uid_mapping[original_uid] = full_new_uid
                        ds[tag].value = f"{uid_root_with_patient}{new_uid}"


def save_uid_mapping(uid_mapping, file_path):
    with open(file_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id_old", "id_new"])
        for original_uid, hashed_uid in uid_mapping.items():
            writer.writerow([original_uid, hashed_uid])


NUMBER1_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
date_pattern = re.compile(r"(19|20)\d{6}")
NAME1_PATTERN = re.compile(r"\bfor [A-Z][a-z]+\s[A-Z][a-z]+\b")
NAME2_PATTERN = re.compile(r"\b[A-Z][a-z]+\s[A-Z][a-z]+ :")
PHONE1_PATTERN = re.compile(r"\d{1}-\d{3}-\d{3}-\d{4}x\d{1}")
PHONE2_PATTERN = re.compile(r"\(\d{3}\)\d{3}-\d{4}x\d{2}")
PHONE3_PATTERN = re.compile(r"\(\d{3}\)\d{3}-\d{4}")
PHONE4_PATTERN = re.compile(r"\d{3}.\d{3}.\d{4}x\d{3}")
PHONE5_PATTERN = re.compile(r"\+\d{1}-\d{3}-\d{3}-\d{4}")
PHONE6_PATTERN = re.compile(r"\b\d{1}-\d{3}-\d{3}-\d{4}\b")
PHONE7_PATTERN = re.compile(r"\d{3}.\d{3}.\d{4}")
PHONE8_PATTERN = re.compile(r"\d{3}-\d{3}-\d{4}")
PHONE9_PATTERN = re.compile(r"call \d{10}")
ABS1_PATTERN = re.compile(r"\bat\s[A-Z]+\b")
ABS2_PATTERN = re.compile(r"\bby\s[A-Z]+\b")
ADDRESS_PATTERN1 = re.compile(r"\d+ [A-Za-z0-9\s]+, [A-Z]{2} \d{5}")
ADDRESS_PATTERN2 = re.compile(r"[A-Za-z0-9\s]+, [A-Z]{2} \d{5}")
HISTORY_PATTERNS = [
    re.compile(p)
    for p in [
        r"\b\w+,\s\w+\sand\s\w+\sMedical Center\b",
        r"\b\w+\sand\s\w+\sMedical Center\b",
        r"\b[A-Z][a-z]+-[A-Z][a-z]+ Medical Center\b",
        r"\b[A-Z][a-z]+ Medical Center\b",
        r"\b\w+,\s\w+\sand\s\w+\sMedical Clinic\b",
        r"\b\w+,\s\w+\sand\s\w+\sGeneral Hospital\b",
        r"\b\w+\sand\s\w+\sGeneral Hospital\b",
        r"\b[A-Z][a-z]+-[A-Z][a-z]+ General Hospital\b",
        r"\b[A-Z][a-z]+ General Hospital\b",
        r"\b\w+,\s\w+\sand\s\w+\sCommunity Clinic\b",
        r"\b\w+\sand\s\w+\sCommunity Clinic\b",
        r"\b[A-Z][a-z]+-[A-Z][a-z]+ Community Clinic\b",
        r"\b\w+,\s\w+\sand\s\w+\sCommunity Hospital\b",
        r"\b\w+\sand\s\w+\sCommunity Hospital\b",
        r"\b[A-Z][a-z]+-[A-Z][a-z]+ Community Hospital\b",
        r"\b\w+,\s\w+\sand\s\w+\sMemorial\b",
        r"\b\w+\sand\s\w+\sMemorial\b",
        r"\b[A-Z][a-z]+-[A-Z][a-z]+ Memorial\b",
        r"\b[A-Z][a-z]+\s[A-Z][a-z]+ Memorial\b",
        r"\b[A-Z][a-z]+ Memorial\b",
        r"\b\w+,\s\w+\sand\s\w+\sGeneral\b",
        r"\b\w+\sand\s\w+\sGeneral\b",
        r"\b[A-Z][a-z]+-[A-Z][a-z]+ General\b",
        r"\b[A-Z][a-z]+ General\b",
        r"\b\w+,\s\w+\sand\s\w+\sClinic\b",
        r"\b[A-Z][a-z]+\s[A-Z][a-z]+ Clinic\b",
        r"\b[A-Z][a-z]+-[A-Z][a-z]+ Clinic\b",
        r"\b[A-Z][a-z]+ Clinic\b",
        r"\b\w+,\s\w+\sand\s\w+\sHospital\b",
        r"\b\w+\sand\s\w+\sHospital\b",
        r"\b[A-Z][a-z]+\s[A-Z][a-z]+ Hospital\b",
        r"\b[A-Z][a-z]+-[A-Z][a-z]+ Hospital\b",
        r"\b[A-Z][a-z]+ Hospital\b",
    ]
]
DOCTOR_PATTERNS = [
    re.compile(p) for p in [r"\bDr\.\s[A-Z][a-z]+\b", r"\bDR\.[A-Z]+\b", r"DR_[A-Z]+"]
]
str_to_remove = [
    "% green/purple high",
    "% white high",
    "3792105090",
    "2929551111",
    "5932656543",
]


def remove_sensitive_info_from_value(value, patient_id):
    if isinstance(value, str):
        if value in ("AL", "CW", "CH", "RD", "RS", "DL", "KM"):
            value = ""
        if patient_id:
            value = value.replace(patient_id, "")
        value = ADDRESS_PATTERN1.sub("", value)
        value = ADDRESS_PATTERN2.sub("", value)
        for p in HISTORY_PATTERNS:
            value = p.sub("", value)
        value = NUMBER1_PATTERN.sub("", value)
        value = date_pattern.sub("", value)
        value = NAME1_PATTERN.sub("for", value)
        value = NAME2_PATTERN.sub(":", value)
        value = PHONE1_PATTERN.sub("", value)
        value = PHONE2_PATTERN.sub("", value)
        value = PHONE3_PATTERN.sub("", value)
        value = PHONE4_PATTERN.sub("", value)
        value = PHONE5_PATTERN.sub("", value)
        value = PHONE6_PATTERN.sub("", value)
        value = PHONE7_PATTERN.sub("", value)
        value = PHONE8_PATTERN.sub("", value)
        value = PHONE9_PATTERN.sub("call", value)
        for p in DOCTOR_PATTERNS:
            value = p.sub("", value)
        value = ABS1_PATTERN.sub("at", value)
        value = ABS2_PATTERN.sub("by", value)
        for s in str_to_remove:
            value = value.replace(s, "")
    elif isinstance(value, IS):
        value = str(value)
        value = remove_sensitive_info_from_value(value, patient_id)
    return value


def remove_sensitive_info(ds, sensitive_tags, patient_id):
    def clear_value(value):
        if isinstance(value, Sequence):
            for item in value:
                if isinstance(item, Dataset):
                    remove_sensitive_info(item, sensitive_tags, patient_id)
        elif isinstance(value, Dataset):
            remove_sensitive_info(value, sensitive_tags, patient_id)
        elif isinstance(value, (str, IS, list)):
            return remove_sensitive_info_from_value(value, patient_id)
        return value

    for tag in sensitive_tags:
        tag = pydicom.tag.Tag(tag)
        if tag in ds:
            ds[tag].value = clear_value(ds[tag].value)


def deid_dicom_directory(
    input_directory,
    output_directory,
    tags_to_clear,
    id_lookup_file,
    shift_csv_path,
    tags_to_shift,
    tags_to_hash,
    uid_root,
    tags_to_add,
    uid_mapping_output_path,
):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    uid_mapping = {}
    id_lookup = {}
    try:
        with open(id_lookup_file, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for line in reader:
                if len(line) == 2:
                    original_id, new_id = line
                    id_lookup[original_id] = new_id
    except FileNotFoundError:
        print(
            f"Warning: ID lookup file not found at {id_lookup_file}. Patient IDs will not be mapped."
        )
    patient_offsets = load_offsets_from_csv(shift_csv_path)
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(".dcm"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_directory)
                output_path = os.path.join(output_directory, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                ds = pydicom.dcmread(input_path, force=True)
                patient_id = ds.PatientID if "PatientID" in ds else None
                remove_sensitive_info(ds, sensitive_tags, patient_id)
                apply_id_mapping(ds, id_lookup)
                modify_patient_name_to_id(ds)
                current_patient_id = (
                    ds.PatientID if "PatientID" in ds else "UnknownPatientID"
                )
                uid_root_prefix = f"1.2.397.4.5."
                clear_tags(ds, tags_to_clear)
                shift_dates(ds, tags_to_shift, patient_offsets)
                if ds.file_meta and "MediaStorageSOPInstanceUID" in ds.file_meta:
                    original_uid = ds.file_meta.MediaStorageSOPInstanceUID
                    new_uid = hash_uid(original_uid)
                    ds.file_meta.MediaStorageSOPInstanceUID = (
                        f"{uid_root_prefix}{current_patient_id}.8.117.{new_uid}"
                    )
                    uid_mapping[original_uid] = ds.file_meta.MediaStorageSOPInstanceUID
                hash_sequence(
                    ds, tags_to_hash, uid_root_prefix, uid_mapping, current_patient_id
                )
                add_missing_tags(ds, tags_to_add)
                set_empty_tags_to_add(ds, tag_not_null)
                ds.save_as(output_path)
                print(f"Processed file saved: {output_path}")
    save_uid_mapping(uid_mapping, uid_mapping_output_path)


if __name__ == "__main__":
    input_dir = os.environ.get("INPUT_DIR")
    output_dir = os.environ.get("OUTPUT_DIR")
    config_dir = os.environ.get("CONFIG_DIR")
    id_lookup_file = os.path.join(config_dir, "patient_id_mapping.csv")
    shift_csv_path = os.path.join(config_dir, "shift.csv")
    uid_mapping_output_file = os.path.join(output_dir, "uid_mapping.csv")
    uid_root = "1.2.375.4.5."
    print(
        f"Input Directory: {input_dir}\nOutput Directory: {output_dir}\nConfig Directory: {config_dir}"
    )
    if not input_dir or not os.path.isdir(input_dir) or not os.listdir(input_dir):
        print("\nERROR: Input directory is empty or does not exist.")
        exit(1)
    deid_dicom_directory(
        input_dir,
        output_dir,
        tags_to_clear,
        id_lookup_file,
        shift_csv_path,
        tags_to_shift,
        tags_to_hash,
        uid_root,
        tag_to_add,
        uid_mapping_output_file,
    )

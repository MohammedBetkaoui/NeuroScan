"""Script simple pour backfiller patient_age/patient_gender dans la collection analyses
Usage: python scripts/backfill_analyses.py
"""
from database.mongodb_connector import get_mongodb
from bson.objectid import ObjectId
from datetime import datetime


def compute_age_from_dob(dob):
    try:
        if not dob:
            return None
        if isinstance(dob, str):
            try:
                dob_dt = datetime.fromisoformat(dob)
            except Exception:
                dob_dt = datetime.strptime(dob, '%Y-%m-%d')
        elif isinstance(dob, datetime):
            dob_dt = dob
        else:
            return None
        today = datetime.now().date()
        birth = dob_dt.date()
        age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        return int(age)
    except Exception as e:
        print(f"Failed to compute age for dob={dob}: {e}")
        return None


def main():
    db = get_mongodb()
    analyses = db.analyses
    patients = db.patients

    query = {'$or': [{'patient_age': {'$exists': False}}, {'patient_age': None}, {'patient_age': 'N/A'}]}
    to_update = list(analyses.find(query))
    print(f"Found {len(to_update)} analyses to backfill")

    updated = 0
    for a in to_update:
        pid = a.get('patient_id')
        if not pid:
            continue
        patient_doc = patients.find_one({'patient_id': pid}) or None
        if not patient_doc and isinstance(pid, str) and len(pid) == 24:
            try:
                patient_doc = patients.find_one({'_id': ObjectId(pid)})
            except Exception:
                patient_doc = None
        if not patient_doc:
            continue
        dob = patient_doc.get('date_of_birth')
        age = compute_age_from_dob(dob)
        gender = patient_doc.get('gender')
        update = {}
        if age is not None:
            update['patient_age'] = age
        if gender:
            update['patient_gender'] = gender
        if update:
            analyses.update_one({'_id': a['_id']}, {'$set': update})
            updated += 1

    print(f"Backfilled {updated} analyses")


if __name__ == '__main__':
    main()

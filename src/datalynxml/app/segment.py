import logging
import requests
import json
import os
import base64

logger = logging.getLogger(__name__)

segment_base_url = "https://api.segment.io/v1/"


def create_auth_header():
    auth_str = f"{os.environ.get('SEGMENT_WRITE_KEY')}:"
    encoded_segment_key = base64.b64encode(auth_str.encode("utf-8")).decode()
    return {
        "Content-Type": "application/json",
        "Authorization": f"Basic {encoded_segment_key}",
    }


def track(user_id, event_name, event_properties=None, context=None):
    url = segment_base_url + "track"

    data = {
        "userId": user_id,
        "event": event_name,
    }

    if event_properties:
        data["properties"] = event_properties

    if context:
        data["context"] = context

    json_data = json.dumps(data)
    headers = create_auth_header()

    response = requests.post(url, headers=headers, data=json_data)

    logger.debug(response.text)
    if response.status_code == 200:
        logger.debug("segment_track: success")
    else:
        logger.error("segment_track: failed", response.text)

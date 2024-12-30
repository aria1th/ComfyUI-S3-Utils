
from .install import initialization

initialization()

from .s3_utils import CLASS_MAPPINGS as S3Mapping, CLASS_NAMES as S3Names


NODE_CLASS_MAPPINGS = {
}
NODE_CLASS_MAPPINGS.update(S3Mapping)


NODE_DISPLAY_NAME_MAPPINGS = {

}
NODE_DISPLAY_NAME_MAPPINGS.update(S3Names)


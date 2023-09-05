from operator import itemgetter

from lena.data.constants.enums.element_types_enum import ElementTypesEnum
from lena.data.constants.enums.scroll_direction_enum import ScrollDirectionEnum
from lena.elements.elements_utils.displacement_features import DisplacementFeatures
from lena.elements.elements_utils.preprocessing.table_preprocessing import TablePreprocessing
from lena.elements.elements_utils.stitching_features import StitchingFeatures
from lena.elements.objects.screenshot_element import ImageSourceObject
from lena.elements.elements_utils.scroll_actions_features import ScrollActionsFeatures
from lena.elements.objects.roi_element import RoiElement
from lena.utils import general_helpers
from lena.elements.objects.table.table_element import TableElement
from lena.utils.config_manager import ConfigManager
from lena.utils.image_processing import filters_helper, contours_helper, morphological_helpers
from lena.elements.features.table.table_cells_features import TableCellsFeatures
from lena.elements.features.scroll_element_features import ScrollElementDetectionsFeatures
from lena.utils.objects_tracking.displacement import OrbBfHomographicalDisplacement


#@dataclass
class TableElementFeatures(TablePreprocessing):
    d = (1200, 1200)

    def __init__(self, table_model, common_model, common_element_features, scroll_buttons_patterns, scrollbar_shift_threshold):
        super().__init__()
        self.__tables_model = table_model
        self.__common_model = common_model
        self.__image_source = None

        self.__scroll_element_features = ScrollElementDetectionsFeatures(common_element_features, scroll_buttons_patterns, scrollbar_shift_threshold)
        self.__table_cells_features = TableCellsFeatures()

    @property
    def image_source(self):
        return self.__image_source

    @image_source.setter
    def image_source(self, screenshot: ImageSourceObject):
        print("Setting value")
        self.__image_source = screenshot

    def __blur_table_area(self, table_roi):

        # Do blur for current roi
        updated_screenshot = self.image_source.get_current_image_source().copy()

        updated_screenshot_w = table_roi.get_x() + table_roi.get_w()
        updated_screenshot_h = table_roi.get_y() + table_roi.get_h()

        updated_screenshot[table_roi.get_y(): updated_screenshot_h, table_roi.get_x(): updated_screenshot_w,:] = filters_helper.Blur(updated_screenshot[table_roi.get_y(): updated_screenshot_h, table_roi.get_x(): updated_screenshot_w, :], (99, 99))

        self.image_source.update_current_image_source(updated_screenshot)

    def __prepare_parameters_for_stitching_features(self):
        horizontal_roi_shift = ConfigManager().config.elements_parameters.table.stitching.table_cells["horizontal_roi_shift"]
        horizontal_stitching_shift = ConfigManager().config.elements_parameters.table.stitching.table_cells["horizontal_stitching_shift"]
        vertical_stitching_shift = ConfigManager().config.elements_parameters.table.stitching.table_cells["vertical_stitching_shift"]
        horizontal_border = ConfigManager().config.elements_parameters.table.stitching.displacement_borders["horizontal_border"]
        vertical_border = ConfigManager().config.elements_parameters.table.stitching.displacement_borders["vertical_border"]

        return horizontal_roi_shift, horizontal_stitching_shift, vertical_stitching_shift, horizontal_border, vertical_border

from operator import itemgetter

from karaburma.data.constants.enums.element_types_enum import ElementTypesEnum
from karaburma.data.constants.enums.scroll_direction_enum import ScrollDirectionEnum
from karaburma.elements.elements_utils.displacement_features import DisplacementFeatures
from karaburma.elements.elements_utils.preprocessing.table_preprocessing import TablePreprocessing
from karaburma.elements.elements_utils.stitching_features import StitchingFeatures
from karaburma.elements.objects.screenshot_element import ImageSourceObject
from karaburma.elements.elements_utils.scroll_actions_features import ScrollActionsFeatures
from karaburma.elements.objects.roi_element import RoiElement
from karaburma.elements.objects.table.table_element import TableElement
from karaburma.utils.config_manager import ConfigManager
from karaburma.utils.image_processing import filters_helper
from karaburma.elements.features.table.table_cells_features import TableCellsFeatures
from karaburma.elements.features.scroll_element_features import ScrollElementDetectionsFeatures
from karaburma.utils.objects_tracking.displacement import OrbBfHomographicalDisplacement


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
        self.__image_source = screenshot

    def __blur_table_area(self, table_roi):
        # Do blur for current roi
        updated_screenshot = self.image_source.get_current_image_source().copy()

        updated_screenshot_w = table_roi.get_x() + table_roi.get_w()
        updated_screenshot_h = table_roi.get_y() + table_roi.get_h()

        updated_screenshot[table_roi.get_y(): updated_screenshot_h, table_roi.get_x(): updated_screenshot_w,:] = filters_helper.blur(updated_screenshot[table_roi.get_y(): updated_screenshot_h, table_roi.get_x(): updated_screenshot_w, :], (99, 99))

        self.image_source.update_current_image_source(updated_screenshot)

    def __prepare_parameters_for_stitching_features(self):
        horizontal_roi_shift = ConfigManager().config.elements_parameters.table.stitching.table_cells["horizontal_roi_shift"]
        horizontal_stitching_shift = ConfigManager().config.elements_parameters.table.stitching.table_cells["horizontal_stitching_shift"]
        vertical_stitching_shift = ConfigManager().config.elements_parameters.table.stitching.table_cells["vertical_stitching_shift"]
        horizontal_border = ConfigManager().config.elements_parameters.table.stitching.displacement_borders["horizontal_border"]
        vertical_border = ConfigManager().config.elements_parameters.table.stitching.displacement_borders["vertical_border"]

        return horizontal_roi_shift, horizontal_stitching_shift, vertical_stitching_shift, horizontal_border, vertical_border

    def __prepare_stitching_features(self, desired_table, desired_table_cells_area, direction):
        horizontal_roi_shift, horizontal_stitching_shift, vertical_stitching_shift, horizontal_border, vertical_border \
            = self.__prepare_parameters_for_stitching_features()

        scroll_features = ScrollActionsFeatures(desired_table, desired_table_cells_area, direction)

        displacement_features = DisplacementFeatures(OrbBfHomographicalDisplacement(), scroll_features,
                                                     horizontal_border, vertical_border)

        stitching_features = StitchingFeatures(displacement_features, horizontal_stitching_shift,
                                               horizontal_roi_shift, vertical_stitching_shift)

        return stitching_features

    def __find_table_on_extended_table(self, table_roi_element, read_text_from_cells=False):
        table_cells_element = self.__table_cells_features.find_table_cells(table_roi_element)
        current_table_element = TableElement(ElementTypesEnum.table.name, 1.0, table_roi_element, None, None, table_cells_element)

        if read_text_from_cells:
            for cell in current_table_element.get_cells_area_element().get_list_cells():
                cell.read_text_from_cell()

        return current_table_element

    def __set_full_table_to_element(self, desired_table, stitched_table, read_text_from_cells=False):
        stitched_table_roi_element = RoiElement(stitched_table, 0, 0, stitched_table.shape[1], stitched_table.shape[0], "table")
        stitched_table_element = self.__find_table_on_extended_table(stitched_table_roi_element, read_text_from_cells)
        desired_table.set_full_table_area(stitched_table_roi_element, stitched_table_element)

    def find_all_tables(self, image_source, blur_after_searching=True):
        self.image_source = image_source
        tables_groups = super().table_element_classification(self.__tables_model, self.image_source)

        for tables_group in tables_groups:
            best_roi = sorted(tables_group, key=itemgetter(1), reverse=True)[0][0]

            # Find scrolls here
            h_scroll, v_scroll = self.__scroll_element_features.find_scrolls(best_roi)

            # Find cells here
            table_cells_element = self.__table_cells_features.find_table_cells(best_roi)

            # Create table element
            current_table_element = TableElement(ElementTypesEnum.table.name, 1.0, best_roi, h_scroll, v_scroll, table_cells_element)
            self.image_source.add_element(current_table_element)

        if(blur_after_searching):
            for table in self.image_source.get_table_elements():
                self.__blur_table_area(table.get_roi_element())

    def find_table_and_expand(self, table_index: int = 0, read_text_from_cells=False):
        self.find_all_tables(self.image_source, False)

        if len(self.image_source.get_table_elements()) > 0:
            if table_index < len(self.image_source.get_table_elements()):
                desired_table = self.image_source.get_table_elements()[table_index]
                desired_table_cells_area = desired_table.get_cells_area_element()

                if desired_table.get_v_scroll() is not None and desired_table.get_h_scroll() is not None:
                    stitching_features = self.__prepare_stitching_features(desired_table, desired_table_cells_area, ScrollDirectionEnum.RIGHT_DOWN.name)
                    stitched_table = stitching_features.right_down_stitch()
                    self.__set_full_table_to_element(desired_table, stitched_table, read_text_from_cells)

                elif desired_table.get_h_scroll() is not None and desired_table.get_v_scroll() is None:
                    stitching_features = self.__prepare_stitching_features(desired_table, desired_table_cells_area, ScrollDirectionEnum.RIGHT.name)
                    stitched_table = stitching_features.right_down_stitch()
                    self.__set_full_table_to_element(desired_table, stitched_table, read_text_from_cells)

                elif desired_table.get_v_scroll() is not None and desired_table.get_h_scroll() is None:
                    stitching_features = self.__prepare_stitching_features(desired_table, desired_table_cells_area, ScrollDirectionEnum.DOWN.name)
                    stitched_table = stitching_features.right_down_stitch()
                    self.__set_full_table_to_element(desired_table, stitched_table, read_text_from_cells)

                elif desired_table.get_v_scroll() is None and desired_table.get_h_scroll() is None:
                    print("Table doesn't have any scrolls")

            else:
                print("Wrong table index")
        else:
            print("Tables were no found")

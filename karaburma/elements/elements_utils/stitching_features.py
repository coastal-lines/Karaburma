import copy
import numpy as np

from karaburma.data.constants.enums.scroll_direction_enum import ScrollDirectionEnum
from karaburma.utils import files_helper


class StitchingFeatures:
    def __init__(self, displacement_features, horizontal_stiching_shift=None, horizontal_roi_shift=None, vertical_stiching_shift=None):
        self.displacement_features = displacement_features

        self.horizontal_stitching_shift = horizontal_stiching_shift
        self.vertical_stitching_shift = vertical_stiching_shift

        if (horizontal_roi_shift != None):
            self.horizontal_roi_shift = horizontal_roi_shift
            self.displacement_features.scroll_features.nested_element.get_roi_element().update_w(self.horizontal_roi_shift)
            #TODO - try to use delegate here?

        self.x_displacement, self.y_displacement = self.__prepare_displacement_values()

        self.nested_element_w, self.nested_element_h = self.displacement_features.scroll_features.nested_element.get_roi_element().get_shape()
        print("")

    def __prepare_displacement_values(self) -> tuple[int, int]:
        x_displacement, y_displacement = self.displacement_features.try_to_find_displacement()

        if (self.horizontal_stitching_shift != None):
            x_displacement += self.horizontal_stitching_shift

        if (self.vertical_stitching_shift != None):
            y_displacement += self.vertical_stitching_shift

        print("x_displacement", x_displacement,"y_displacement", y_displacement)
        return x_displacement, y_displacement

    def __get_crop_after_scroll(self, temp_nested_element):
        match self.displacement_features.scroll_features.direction:
            case ScrollDirectionEnum.RIGHT.name:
                roi_after = temp_nested_element[:, self.nested_element_w - self.x_displacement: self.nested_element_w, :]
                return roi_after

            case ScrollDirectionEnum.DOWN.name:
                roi_after = temp_nested_element[self.nested_element_h - self.y_displacement: self.nested_element_h, :, :]
                return roi_after

    def right_down_stitch(self):
        list_full_horizontal_rows = []

        self.displacement_features.scroll_features.direction = ScrollDirectionEnum.RIGHT.name
        start_full_roi = self.horizontal_stitch()

        while(self.displacement_features.scroll_features.scroll_element(ScrollDirectionEnum.DOWN.name)[0]):
            self.displacement_features.scroll_features.direction = ScrollDirectionEnum.LEFT.name
            self.displacement_features.scroll_features.scroll_until_its_possible()
            self.displacement_features.scroll_features.direction = ScrollDirectionEnum.RIGHT.name

            #stitched_element, full_roi = self.horizontal_stitch()
            stitched_element = self.horizontal_stitch()
            full_horizontal_row = stitched_element[self.nested_element_h - self.y_displacement:self.nested_element_h, :, :]
            list_full_horizontal_rows.append(full_horizontal_row)

            self.displacement_features.scroll_features.direction = ScrollDirectionEnum.DOWN.name

        updated_h = self.y_displacement * len(list_full_horizontal_rows)
        full_roi = np.ones((updated_h, start_full_roi.shape[1], 3), dtype=np.uint8) * 255
        for i in range(len(list_full_horizontal_rows)):
            full_roi[i * self.y_displacement: (i * self.y_displacement) + self.y_displacement, :, :] = list_full_horizontal_rows[i]

        stitched_element = np.ones((self.nested_element_h + updated_h, start_full_roi.shape[1], 3), dtype=np.uint8) * 255
        stitched_element[0:self.nested_element_h, :, :] = start_full_roi[0:self.nested_element_h, :, :]
        stitched_element[self.nested_element_h:self.nested_element_h + updated_h, :, :] = full_roi

        #files_helper.save_image(stitched_element, "9999full_table")

        return stitched_element

    def vertical_stitch(self):
        list_additional_roi = []

        self.displacement_features.scroll_features.nested_element.get_roi_element().update_element_roi_area_by_screenshot()
        start_roi = copy.copy(self.displacement_features.scroll_features.nested_element.get_roi_element().get_roi())
        #files_helper.save_image(start_roi)

        while(self.displacement_features.scroll_features.scroll_element()[0]):
            self.displacement_features.scroll_features.nested_element.get_roi_element().update_element_roi_area_by_screenshot()
            temp_current_element_roi = self.displacement_features.scroll_features.nested_element.get_roi_element().get_roi()
            temp_crop = self.__get_crop_after_scroll(temp_current_element_roi)
            list_additional_roi.append(temp_crop)

        updated_h = self.y_displacement * len(list_additional_roi)

        #if self.horizontal_roi_shift is not None:
        #    self.nested_element_w += self.horizontal_roi_shift * -1

        full_roi = np.ones((updated_h, self.nested_element_w, 3), dtype=np.uint8) * 255
        for i in range(len(list_additional_roi)):
            full_roi[i * self.y_displacement: (i * self.y_displacement) + self.y_displacement, :, :] = list_additional_roi[i]

        #files_helper.save_image(full_roi, "full_roi")
        #files_helper.save_image(start_roi, "start_roi")

        mega_super_stiched_roi = np.ones((self.nested_element_h + updated_h, self.nested_element_w,  3), dtype=np.uint8) * 255
        mega_super_stiched_roi[0:self.nested_element_h, :, :] = start_roi[0:self.nested_element_h, :, :]
        mega_super_stiched_roi[self.nested_element_h:self.nested_element_h + updated_h, :, :] = full_roi

        return mega_super_stiched_roi

    def horizontal_stitch(self):
        list_additional_roi = []

        #w, h = self.nested_element.get_roi_element().get_shape()

        self.displacement_features.scroll_features.nested_element.get_roi_element().update_element_roi_area_by_screenshot()
        start_roi = copy.copy(self.displacement_features.scroll_features.nested_element.get_roi_element().get_roi())

        while(self.displacement_features.scroll_features.scroll_element()[0]):
            self.displacement_features.scroll_features.nested_element.get_roi_element().update_element_roi_area_by_screenshot()
            temp_current_element_roi = self.displacement_features.scroll_features.nested_element.get_roi_element().get_roi()
            temp_crop = self.__get_crop_after_scroll(temp_current_element_roi)
            list_additional_roi.append(temp_crop)

        updated_w = self.x_displacement * len(list_additional_roi)

        full_roi = np.ones((self.nested_element_h, updated_w, 3), dtype=np.uint8) * 255
        for i in range(len(list_additional_roi)):
            full_roi[:, i * self.x_displacement: (i * self.x_displacement) + self.x_displacement, :] = list_additional_roi[i]

        #files_helper.save_image(full_roi, "mega_super_stiched_full_roi")

        mega_super_stiched_roi = np.ones((self.nested_element_h, self.nested_element_w + updated_w, 3), dtype=np.uint8) * 255
        mega_super_stiched_roi[:, 0:self.nested_element_w, :] = start_roi[:, 0:self.nested_element_w, :]
        mega_super_stiched_roi[:, self.nested_element_w:self.nested_element_w + updated_w, :] = full_roi

        #files_helper.save_image(mega_super_stiched_roi, "mega_super_stiched_roi")

        #return mega_super_stiched_roi, full_roi
        return mega_super_stiched_roi
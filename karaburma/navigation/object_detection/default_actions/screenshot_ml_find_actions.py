from karaburma.navigation.object_detection.default_actions.common_ml_find_actions import CommonMlFindActions


class ScreenshotMlFindActions(CommonMlFindActions):
    def __init__(self, common_element_features, table_element_features, listbox_element_features):
        super().__init__(common_element_features, table_element_features, listbox_element_features)
        self.table_element_features = table_element_features

    def find_table_and_expand(self, image_source, table_index):
        self.table_element_features.image_source = image_source
        self.table_element_features.find_table_and_expand(table_index)

        # DEBUG
        #draw_elements(image_source.get_current_image_source_copy(), image_source)

    def find_table_cell(self, image_source, column, row):
        #TODO
        table = self.find_table_and_expand(image_source)
        #TODO
        #get cell by index
        #APP_DATA.table_features.test_horizontal_sroll_and_stich(screenshot_elements)

    def find_listbox_and_expand(self, image_source, listbox_index):
        self.listbox_element_features.find_listbox_and_expand(image_source, listbox_index)

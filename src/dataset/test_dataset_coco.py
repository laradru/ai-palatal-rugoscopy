import unittest
import tempfile
import shutil
import os
from src.dataset.dataset_coco import CocoDatasetInstanceSegmentation
from src.dataset.dataset_coco import extended_dimensions

class TestCocoDataset(unittest.TestCase):
    def test_extract_patches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f'The path for the output is {temp_dir}')
            dataset = CocoDatasetInstanceSegmentation(data_directory_path='./data/1/images',
                                                      data_annotation_path='./data/1/annotations/instances_default.json')

            output_dir = os.path.join(temp_dir, 'output_patches')
            patch_size = 64 
            stride = 13
            min_area = 100.0
            dataset.extract_patches(output_dir, patch_size, stride, min_area)

            self.assertTrue(os.path.exists(output_dir))
            self.assertTrue(os.path.exists(os.path.join(output_dir, 'images')))
            self.assertTrue(os.path.exists(os.path.join(output_dir, 'annotations')))
            self.assertTrue(os.path.exists(os.path.join(output_dir, 'annotations', 'annotations.json')))
            
    def test_extended_dimensions(self):
        
        assert(extended_dimensions(4, 2, [10, 10]) == [10, 10])
        assert(extended_dimensions(4, 3, [11, 11]) == [13, 13])
        assert(extended_dimensions(4, 2, [12, 12]) == [12, 12])
        assert(extended_dimensions(4, 2, [13, 13]) == [16, 16])
        assert(extended_dimensions(3, 2, [9, 9]) == [9, 9])
        assert(extended_dimensions(3, 1, [10, 10]) == [10, 10])
        
        # For stride=1 there's no need to extend dimensions never.
        assert(extended_dimensions(256, 1, [5000, 5000]) == [5000, 5000])
        assert(extended_dimensions(256, 1, [50000, 50000000]) == [50000, 50000000])
        assert(extended_dimensions(256, 1, [123456, 987654]) == [123456, 987654])
        assert(extended_dimensions(256, 1, [1122554466, 12]) == [1122554466, 12])
        
        # For weird strides, still, the extension should fit the patches.
        assert(extended_dimensions(4, 8, [10, 10]) == [12, 12])
        assert(extended_dimensions(3, 8, [10, 10]) == [11, 11])


if __name__ == '__main__':
    unittest.main()

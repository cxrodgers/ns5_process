import os
import shutil
import glob

class FileCollater:
    def __init__(self, data_dir_list, filenamestring, output_dir,
        output_file_name=None):
        self.data_dir_list = data_dir_list
        self.filenamestring = filenamestring
        self.output_dir = output_dir
        self.output_file_name = output_file_name
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        if self.output_file_name is None:
            self._generate_output_file_name()
    
    def copy_files(self):       
        for n, data_dir in enumerate(self.data_dir_list):
            tgt_name_list = glob.glob(os.path.join(data_dir, 
                self.filenamestring))
            for tgt_name in tgt_name_list:
                dst_name = os.path.join(self.output_dir, 
                    self.output_file_name[n] + os.path.split(tgt_name)[1])
                shutil.copyfile(tgt_name, dst_name)
    
    def _generate_output_file_name(self):
        self.output_file_name = list()
        for data_dir in self.data_dir_list:
            split_data_dir = data_dir.split(os.path.sep)
            self.output_file_name.append(\
                split_data_dir[-2] + '_' + split_data_dir[-1] + '_')
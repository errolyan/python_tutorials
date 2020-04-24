class Properties(object):

    def __init__(self, file_name):
        self.file_name = file_name
        self.properties = {}

    def _get_dict(self, str_name, dict_name, value):
        if (str_name.find('.') > 0):
            k = str_name.split('.')[0]
            dict_name.setdefault(k, {})
            return self._get_dict(str_name[len(k) + 1:], dict_name[k], value)
        else:
            # print(value)
            # self.properties += str_name + "=" + value + ","
            dict_name[str_name] = value
            return

    def get_properties(self):
        try:
            pro_file = open(self.file_name, 'Ur')
            for line in pro_file.readlines():
                line = line.strip().replace('\n', '')
                if line.find("#") != -1:
                    line = line[0:line.find('#')]
                if line.find('=') > 0:
                    strs = line.split('=')
                    strs[1] = line[len(strs[0]) + 1:]
                    self._get_dict(strs[0].strip(), self.properties, strs[1].strip())
        except Exception as e:
            raise e
        else:
            pro_file.close()
        return self.properties

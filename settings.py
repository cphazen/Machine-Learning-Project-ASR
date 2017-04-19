import os

from audio_processing.sinusoid import ImproperlyConfigured


class ProjectSettingsBase(object):

    """
    Base class
    """

    _path_delimiter = "\\"
    _base_path = os.getcwd()
    _custom_settings = {}


    def p(self, relative_path): return os.path.normpath(relative_path)
    _string_dictionary = {}

    def get_string(self, key):
    	if key=="transcript_type": return self._string_dictionary[key]
    	else: return os.path.join(self._base_path, self._string_dictionary[key])

    def _set_string(self, key, string):
    	self._string_dictionary[key] = str(string)

    def __init__(self):

    	_default_dict = {"phone_file": self.p("an4\\etc\\an4.phone"), # path to file of phone list
   	                     "dictionary_file": self.p("an4\\etc\\an4.dic"), # path to file of word list
   	                     "transcript_file": self.p("an4\\etc\\an4_train.transcription"), # path/directory of transcript file(s)
   	                     "transcript_type": "CMU AN4", # library being used (to find & parse transcript file)
   	                     "words_to_phones_file": self.p("an4\\etc\\an4.dic"), # path to file matching words with phones
   	                     "directory_path": self.p("an4\\feat\\an4_clstk")} # directory containing audio files

    	for key in _default_dict:
    		self._set_string(key, _default_dict[key])

    	for key in self._custom_settings:
    		self._set_string(key, self._custom_settings[key])



class ProjectSettingsUFLabComputers(ProjectSettingsBase):
    """
    These settings work on the Windows
    machines at the Hub or in the Libraries.
    """

    _base_path = "C:\\Users\\jr8000\Desktop\\Machine-Learning-Project-ASR-master"
    _custom_settings = {"example_setting_key_1": "example_setting_string_1",
                        "example_setting_key_2": "example_setting_string_2"}


class ProjectSettingsForSomeOtherWorkStation(ProjectSettingsBase):
    """
    Just stub code for running the
    project on a different device.
    """

    #_base_path = "..."
    #_custom_settingss = {"...": "..."}
    pass



configuration = ProjectSettingsUFLabComputers()

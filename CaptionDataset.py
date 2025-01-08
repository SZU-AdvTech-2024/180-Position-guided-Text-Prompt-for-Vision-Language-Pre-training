import json

class CaptionDataset:
    def __init__(self):

        self.data = {}

    def add_data(self, imageid, caption):
        if imageid not in self.data:
            self.data[imageid] = [0, []]
        self.data[imageid][1].append(caption)

    def get_next_caption(self, imageid):

        if imageid not in self.data:
            return ''
        
        index, captions = self.data[imageid]

        if not captions:
            return ''

        caption = captions[index]

        self.data[imageid][0] = (index + 1) % len(captions)

        return caption
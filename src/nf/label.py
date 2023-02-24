
class Labeler:

    def __init__(self):
        label_id_map = {k: v for v, k in enumerate(const_ner_tags) if k not in self.remove_ner_tags}
        self.label_id_map = label_id_map
        ids_to_labels = {v: k for v, k in enumerate(const_ner_tags) if k not in self.remove_ner_tags}
        self.ids_to_labels = ids_to_labels

    def ids(self):
        pass
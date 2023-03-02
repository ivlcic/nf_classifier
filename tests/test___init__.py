import os
from unittest import TestCase

from nf import MultiLabeler


class TestMultiLabeler(TestCase):

    def test_init(self):
        labeler: MultiLabeler = MultiLabeler(labels=['Economy', 'Labour', 'Welfare', 'Security', 'Culture'])
        self.assertEqual(['',
                          'Economy',
                          'Labour',
                          'EconomyLabour',
                          'Welfare',
                          'EconomyWelfare',
                          'LabourWelfare',
                          'EconomyLabourWelfare',
                          'Security',
                          'EconomySecurity',
                          'LabourSecurity',
                          'EconomyLabourSecurity',
                          'WelfareSecurity',
                          'EconomyWelfareSecurity',
                          'LabourWelfareSecurity',
                          'EconomyLabourWelfareSecurity',
                          'Culture',
                          'EconomyCulture',
                          'LabourCulture',
                          'EconomyLabourCulture',
                          'WelfareCulture',
                          'EconomyWelfareCulture',
                          'LabourWelfareCulture',
                          'EconomyLabourWelfareCulture',
                          'SecurityCulture',
                          'EconomySecurityCulture',
                          'LabourSecurityCulture',
                          'EconomyLabourSecurityCulture',
                          'WelfareSecurityCulture',
                          'EconomyWelfareSecurityCulture',
                          'LabourWelfareSecurityCulture',
                          'EconomyLabourWelfareSecurityCulture'], labeler.labels)

    def test_decode(self):
        labeler: MultiLabeler = MultiLabeler(labels=['Economy', 'Labour', 'Welfare', 'Security', 'Culture'])
        labels = labeler.decode(18)
        self.assertEqual(['Labour', 'Culture'], labels)

        labels = labeler.decode(19)
        self.assertEqual(['Economy', 'Labour', 'Culture'], labels)

        labels = labeler.decode(20)
        self.assertEqual(['Welfare', 'Culture'], labels)

    def test_binpowset(self):
        labeler: MultiLabeler = MultiLabeler(labels=['Economy', 'Labour', 'Welfare', 'Security', 'Culture'])
        labels = labeler.binpowset(18)
        self.assertEqual([0, 1, 0, 0, 1], labels)

        labels = labeler.binpowset(19)
        self.assertEqual([1, 1, 0, 0, 1], labels)

        labels = labeler.binpowset(20)
        self.assertEqual([0, 0, 1, 0, 1], labels)

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

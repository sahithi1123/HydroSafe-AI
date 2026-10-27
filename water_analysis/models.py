from django.db import models
from django.contrib.auth.models import User


class AnalysisReport(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="analysis_reports"
    )

    water_parameters = models.JSONField()
    ml_prediction = models.CharField(max_length=50)
    dl_results = models.JSONField(default=list)
    fusion_result = models.JSONField(default=dict)
    ai_report = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Report #{self.id} - {self.user.username}"

    class Meta:
        ordering = ["-created_at"]
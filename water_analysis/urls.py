from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),

    # Authentication
    path("register/", views.register, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),

    # Water analysis
    path("analysis/", views.analysis, name="analysis"),
    path(
        "microscopic-analysis/",
        views.microscopic_analysis,
        name="microscopic_analysis"
    ),

    # Reports
    path("final-report/", views.final_report, name="final_report"),
    path(
        "generate-ai-report/",
        views.generate_ai_report,
        name="generate_ai_report"
    ),
    path(
        "download-report/",
        views.download_report,
        name="download_report"
    ),
    path("reports/", views.reports, name="reports"),
]
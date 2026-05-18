from setuptools import setup, find_packages

setup(
    name="bioprocess_dashboard",
    version="0.1.0",
    description="MSPC Dashboard for Biologics Manufacturing Process Monitoring",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "shap>=0.43.0",
        "dash>=2.14.0",
        "dash-bootstrap-components>=1.5.0",
        "plotly>=5.17.0",
        "sqlalchemy>=2.0.0",
        "pyyaml>=6.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "jupyter>=1.0.0"],
    },
)

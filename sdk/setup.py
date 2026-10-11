from setuptools import setup, find_packages

setup(
    name="aitrajectiq",
    version="0.1.0",
    description="SDK d'observabilité pour applications LLM — capture transparente des appels LLM et outils",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[],
    entry_points={
        "console_scripts": [
            "aitrajectiq=aitrajectiq.__main__:main",
            "aitrajectiq-run=aitrajectiq.cli:main",
        ],
    },
    extras_require={
        "openai": ["openai>=1.0"],
        "anthropic": ["anthropic>=0.18"],
        "mistral": ["mistralai>=0.1"],
        "langchain": ["langchain-core>=0.1"],
        "all": ["openai>=1.0", "anthropic>=0.18", "mistralai>=0.1", "langchain-core>=0.1"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: System :: Monitoring",
    ],
)

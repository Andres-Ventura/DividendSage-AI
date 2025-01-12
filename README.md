# DividendSage-AI

## Overview
DividendSage AI is a sophisticated platform designed to track and analyze the dividends of US large public companies using cutting-edge artificial intelligence. The system leverages advanced machine learning models, robust data pipelines, and seamless integrations to provide actionable insights into dividend trends, payment schedules, and forecasts.

## Key Features
- **Dividend Forecasting**: AI-powered models to predict dividend payouts and trends.
- **Quarterly Tracking**: Automated monitoring of dividend schedules for US large public companies.
- **Data Pipelines**: Efficient ETL processes to extract, clean, and load financial data.
- **REST API**: Exposes data and insights through secure API endpoints.
- **Infrastructure Automation**: Scalable infrastructure with Terraform, Kubernetes, and CI/CD pipelines.
- **Monitoring and Alerts**: Real-time monitoring with predefined Grafana dashboards and alerting systems.

---

## File Structure

### `docs/`
Documentation for developers and stakeholders.
- `requirements.md`: Functional specifications.
- `architecture-diagram.png`: High-level and low-level architectural diagrams.
- `ADR/`: Records key design decisions.
- `devops-setup.md`: Details about DevOps workflows and conventions.

### `src/`
Application source code.
- **AI**: AI models and pipelines for dividend predictions.
  - `models/`: Model implementations (e.g., `dividend_forecast.py`).
  - `pipelines/`: Training and deployment pipelines.
  - `orchestration/`: Workflow orchestration logic.
- **Core**: Core business logic and domain-specific services.
  - `entities.py`: Domain entities.
  - `services.py`: Business logic services.
- **Data**: Data management and ETL pipelines.
  - `etl_pipeline.py`: Extract-transform-load pipeline logic.
- **API**: REST API to expose functionality.
  - `app.py`: API entry point.
  - `routes/`: Endpoint definitions (e.g., `dividends.py`).
- **DB**: Database interactions and migrations.
- **Utils**: Shared utilities for logging, configuration, and constants.

### `tests/`
Comprehensive test suite.
- `unit/`: Unit tests.
- `integration/`: Integration tests.
- `e2e/`: End-to-end tests.
- `performance/`: Stress and performance tests.
- `test_setup/`: Shared test fixtures.

### `infra/`
Infrastructure as code.
- `terraform/`: Modular configurations for cloud resources.
- `kubernetes/`: K8s manifests and Helm charts.
- `ansible/`: Playbooks for server provisioning.

### `monitoring/`
Observability tools and configurations.
- `alerts/`: Alert definitions.
- `dashboards/`: Predefined Grafana dashboards.
- `logs/`: Logging configurations.

### Additional Files
- `Makefile`: Commands for building, testing, and deploying the system.
- `pyproject.toml`: Python dependency management.
- `.env`: Environment variable configuration.
- `README.md`: Project overview and setup instructions (this file).
- `LICENSE`: License for the project.

---

## Getting Started

### Prerequisites
1. **Python 3.10+**: For application runtime.
2. **Docker**: To containerize and run services.
3. **Terraform & Kubectl**: For infrastructure management.
4. **Node.js**: For additional build tools if required.
5. **PostgreSQL**: Backend database.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/financial-system-ai.git
   cd financial-system-ai
   ```

2. Install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements/base.txt
   ```

3. Set up the environment variables:
   ```bash
   cp .env.example .env
   # Update .env with your settings
   ```

4. Run the application:
   ```bash
   docker-compose up --build
   ```

### Running Tests
Execute the test suite:
```bash
make test
```

---

## Deployment
1. Configure Terraform for infrastructure provisioning:
   ```bash
   cd infra/terraform/prod
   terraform init
   terraform apply
   ```

2. Deploy Kubernetes manifests:
   ```bash
   kubectl apply -k infra/kubernetes/overlays/prod
   ```

3. Monitor deployments via Grafana dashboards and logs.

---

## Contributions
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes with a descriptive message.
4. Push to your branch and open a Pull Request.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Contact
For questions or support, reach out to [andres.venturase@gmail.com].


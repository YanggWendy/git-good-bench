"""Summarize results written by run_cli_agent_benchmark.py.

    python -m src.agent_client.summarize_cli_agent_results results/<run-name>/results.jsonl [...]

Prints the solve rate per run (mean over samples), the mean and standard
deviation over runs, a per-language breakdown and the number of errors.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def summarize(results_file: Path) -> None:
    # With --retry-errors a (sample, scenario, run) triple can appear more than once; the last record counts.
    latest = {}
    for line in results_file.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            record.setdefault('scenario_type', 'merge')
            latest[(record['sample_id'], record['scenario_type'], record['run'])] = record
    if not latest:
        print(f'{results_file}: no results')
        return
    for scenario_type in sorted({record['scenario_type'] for record in latest.values()}):
        summarize_scenario(results_file, scenario_type,
                           [record for record in latest.values() if record['scenario_type'] == scenario_type])


def summarize_scenario(results_file: Path, scenario_type: str, records: list) -> None:
    runs = sorted({record['run'] for record in records})
    samples = sorted({record['sample_id'] for record in records})
    print(f'{results_file} [{scenario_type}]: {len(samples)} samples, runs {runs}, agent={records[0]["agent"]} '
          f'model={records[0].get("model")}')

    rates = []
    for run in runs:
        run_records = [record for record in records if record['run'] == run]
        solved = sum(record['solved'] for record in run_records)
        errors = sum(record['status'] != 'evaluated' for record in run_records)
        timeouts = sum(bool(record.get('agent_timed_out')) for record in run_records)
        rate = 100 * solved / len(run_records)
        rates.append(rate)
        print(f'  run {run}: {solved}/{len(run_records)} solved = {rate:.2f}% '
              f'({errors} not evaluated, {timeouts} agent timeouts)')
    if len(rates) > 1:
        print(f'  mean {statistics.mean(rates):.2f}% +- {statistics.stdev(rates):.2f}% (std over runs)')

    by_language = defaultdict(list)
    for record in records:
        by_language[record['programming_language']].append(record['solved'])
    for language, solved in sorted(by_language.items()):
        print(f'  {language}: {sum(solved)}/{len(solved)} = {100 * sum(solved) / len(solved):.2f}%')

    errors = [record for record in records if record['status'] != 'evaluated']
    for record in errors:
        print(f'  ERROR {record["sample_id"]} run {record["run"]}: {record.get("error")}')


def main() -> None:
    for path in sys.argv[1:]:
        summarize(Path(path))


if __name__ == '__main__':
    main()

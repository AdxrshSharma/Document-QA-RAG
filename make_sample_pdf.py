"""Creates data/acme_handbook.pdf: a 3-page demo document so you can try the project
immediately. Afterwards, drop your OWN PDFs into data/ for a stronger portfolio piece."""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

PAGES = [
    ("Acme Corp Employee Handbook - Leave Policy", [
        "Every full-time employee receives 24 days of paid annual leave per calendar year. "
        "Leave accrues monthly and can be requested through the HR portal at least one week in advance.",
        "Employees are entitled to 12 days of paid sick leave each year. A medical certificate is "
        "required for sick leave longer than 3 consecutive days.",
        "Up to 10 days of unused annual leave can be carried forward to the next year. Any leave "
        "beyond that limit expires on 31 March.",
        "Maternity leave is 26 weeks of fully paid leave. Paternity leave is 2 weeks of fully paid leave.",
    ]),
    ("Acme Corp Employee Handbook - Work Hours and Benefits", [
        "Acme follows a hybrid model: employees work from the office 3 days per week and from home "
        "for the remaining days. Core working hours are 10 AM to 4 PM on all working days.",
        "Employees working remotely receive an internet allowance of 1,000 INR per month.",
        "Company laptops are refreshed every 3 years. Employees may request an earlier replacement "
        "if their device becomes unusable.",
        "Each employee has a learning budget of 50,000 INR per year for courses, books and conferences. "
        "Unused budget does not roll over.",
    ]),
    ("Acme Corp Employee Handbook - Performance and Policies", [
        "Performance reviews take place twice a year, in June and December. Each review includes a "
        "self-assessment and a manager assessment.",
        "The promotion cycle runs once a year in April and is based on the last two performance reviews.",
        "The standard notice period for resignation is 60 days. Employees in their probation period "
        "have a notice period of 15 days.",
        "Concerns about workplace conduct can be reported confidentially to hr-help@acme.example at any time.",
    ]),
]


def main():
    Path("data").mkdir(exist_ok=True)
    c = canvas.Canvas("data/acme_handbook.pdf", pagesize=A4)
    width, height = A4
    for title, paragraphs in PAGES:
        y = height - 70
        c.setFont("Helvetica-Bold", 16)
        c.drawString(60, y, title)
        y -= 35
        c.setFont("Helvetica", 11)
        for p in paragraphs:
            for line in simpleSplit(p, "Helvetica", 11, width - 120):
                c.drawString(60, y, line)
                y -= 16
            y -= 10
        c.showPage()
    c.save()
    print("Created data/acme_handbook.pdf (3 pages)")


if __name__ == "__main__":
    main()

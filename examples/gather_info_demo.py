"""
Gather Info Mode Demo

Demonstrates the contexts system's gather_info mode for structured data
collection. Unlike the InfoGathererAgent prefab, this uses the low-level
contexts API with set_gather_info() and add_gather_question().

Gather info mode presents questions one at a time, producing zero
tool_call/tool_result entries in conversation history. Answers are
stored in global_data under the configured output key.
"""

from signalwire_agents import AgentBase


class PatientIntakeAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="Patient Intake Agent",
            route="/patient-intake"
        )

        self.add_language("English", "en-US", "inworld.Mark")

        self.prompt_add_section(
            "Role",
            "You are a friendly medical office intake assistant. "
            "Collect patient information accurately and professionally."
        )

        # Define a context with gather info steps
        ctx_builder = self.define_contexts()
        ctx = ctx_builder.add_context("default")

        # Step 1: Gather patient demographics
        step1 = ctx.add_step("demographics")
        step1.set_text("Collect the patient's basic information.")
        step1.set_gather_info(
            output_key="patient_demographics",
            prompt="Please collect the following patient information."
        )
        step1.add_gather_question("full_name", "What is your full name?", type="string")
        step1.add_gather_question("date_of_birth", "What is your date of birth?", type="string")
        step1.add_gather_question("phone_number", "What is your phone number?", type="string", confirm=True)
        step1.add_gather_question("email", "What is your email address?", type="string")
        step1.set_valid_steps(["symptoms"])

        # Step 2: Gather symptoms
        step2 = ctx.add_step("symptoms")
        step2.set_text("Ask about the patient's current symptoms and reason for visit.")
        step2.set_gather_info(
            output_key="patient_symptoms",
            prompt="Now let's talk about why you're visiting today."
        )
        step2.add_gather_question("reason_for_visit", "What is the main reason for your visit today?", type="string")
        step2.add_gather_question("symptom_duration", "How long have you been experiencing these symptoms?", type="string")
        step2.add_gather_question("pain_level", "On a scale of 1 to 10, how would you rate your discomfort?", type="string")
        step2.set_valid_steps(["confirmation"])

        # Step 3: Confirmation (normal mode, not gather)
        step3 = ctx.add_step("confirmation")
        step3.set_text(
            "Summarize all the information collected and confirm with the patient "
            "that everything is correct. Thank them for their time."
        )
        step3.set_step_criteria("Patient has confirmed all information is correct")


if __name__ == "__main__":
    agent = PatientIntakeAgent()
    agent.run()

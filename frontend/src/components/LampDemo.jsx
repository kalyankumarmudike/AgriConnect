import React from "react";
import { motion } from "framer-motion";
import farming from "../assets/farming2.jpg"; // Adjust path if needed

export default function LampDemo() {
  return (
    <LampContainer>
      <motion.div
        initial={{ opacity: 0.5, y: 100 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{
          delay: 0.3,
          duration: 0.8,
          ease: "easeInOut",
        }}
        className="mt-8 relative z-30"
      >
        <img
          src={farming}
          alt="Farming"
          className="rounded-xl w-full max-w-xl"
        />
      </motion.div>
    </LampContainer>
  );
}

const LampContainer = ({ children, className }) => {
  return (
    <div
      className={`relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-transparent w-full rounded-md z-0 ${className}`}
    >
      {/* Children (Image) — translation fixed here */}
      <div className="relative z-50 flex flex-col items-center px-5">
        {children}
      </div>
    </div>
  );
};
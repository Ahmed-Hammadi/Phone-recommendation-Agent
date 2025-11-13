import type { Config } from "tailwindcss";
import plugin from "tailwindcss/plugin";

const config: Config = {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"]
,  theme: {
    extend: {
      colors: {
        midnight: {
          50: "#f1f4ff",
          100: "#dfe6ff",
          200: "#bcc8ff",
          300: "#8b9cff",
          400: "#4a61ff",
          500: "#1d2bff",
          600: "#0414f1",
          700: "#020fcc",
          800: "#040f9d",
          900: "#060e7f",
          950: "#04084c",
        },
      },
      backgroundImage: {
        "glass-gradient": "linear-gradient(135deg, rgba(69, 95, 199, 0.25), rgba(23, 57, 144, 0.05))",
        "neon-grid": "radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.08) 0, rgba(255, 255, 255, 0) 40%), radial-gradient(circle at 80% 20%, rgba(255, 255, 255, 0.08) 0, rgba(255, 255, 255, 0) 40%)",
      },
      boxShadow: {
        glow: "0 0 40px rgba(99, 102, 241, 0.35)",
      },
      blur: {
        xs: "2px",
      },
    },
  },
  plugins: [
    plugin(({ addUtilities }) => {
      addUtilities({
        ".glass-panel": {
          backdropFilter: "blur(14px)",
          backgroundColor: "rgba(15, 23, 42, 0.25)",
          border: "1px solid rgba(148, 163, 239, 0.3)",
          boxShadow: "0 25px 45px rgba(15, 23, 42, 0.35)",
        },
        ".glass-panel-light": {
          backdropFilter: "blur(12px)",
          backgroundColor: "rgba(255, 255, 255, 0.55)",
          border: "1px solid rgba(148, 163, 239, 0.3)",
          boxShadow: "0 20px 40px rgba(15, 23, 42, 0.2)",
        },
      });
    }),
  ],
};

export default config;
